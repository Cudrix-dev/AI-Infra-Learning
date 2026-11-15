"""Day 12 - shared pieces: problem, stage split, loss weighting, Megatron-style P2P helpers, the per-rank event log.

Everything that must be IDENTICAL between the single-GPU baseline and the two-stage pipeline lives here:
  * make_problem(): input x, target and every block's weights, drawn on the CPU from one seed in float64, then cast;
  * Block: a residual MLP block, out = x + fc2(gelu(fc1(x))), activations [seq, batch, d] as in Megatron and Day 11;
  * stage_blocks(): which blocks each stage owns (4 blocks, 2 stages: stage 0 = blk0, blk1; stage 1 = blk2, blk3);
  * microbatches(): split the effective batch along dim 1 (batch) into M contiguous chunks;
  * mb_loss(): one microbatch's share of the full-batch mean loss (Day 04's n_m / N weight).

Provided for the pipeline (you do NOT write these; they mirror Megatron's names so Day 13's reading maps 1:1):
  * P2P: recv_forward, send_forward, recv_backward, send_backward, send_forward_recv_backward,
    send_backward_recv_forward. A call that both sends and receives posts the two ops together with
    dist.batch_isend_irecv, like Megatron's _communicate with batch_p2p_comm=True.
  * Stage: what one rank knows (its module, its data or targets, its P2P helper, the logged step functions).
  * EventLog: every forward/backward/P2P call each rank makes, in order, with GPU timestamps and the in-flight count.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / "outputs" / "day12_pp"          # JSON results and event logs (archived by the runner)
TENSORS = OUT / "tensors"                       # baseline .pt files (NOT archived: ~200 MB, reproducible from the seed)

DTYPES = {"float32": torch.float32, "float64": torch.float64}
# Pass criterion for every compared tensor: max|pp - ref| / max|ref| <= REL_TOL (relative to the tensor's own scale).
REL_TOL = {torch.float32: 1e-5, torch.float64: 1e-12}


@dataclass(frozen=True)
class Shapes:
    seq: int = 256
    batch: int = 8          # EFFECTIVE batch (sequences per step); split into M microbatches of batch / M
    d_model: int = 512
    ffn: int = 2048
    blocks: int = 4         # residual MLP blocks, split evenly over the stages
    stages: int = 2


def set_exact_math() -> None:
    """FP32 matmuls without TF32, so FP32 differences are rounding-order only."""
    torch.set_float32_matmul_precision("highest")


def make_problem(shapes: Shapes, dtype: torch.dtype, seed: int = 1234) -> dict:
    g = torch.Generator().manual_seed(seed)
    s, b, d, f = shapes.seq, shapes.batch, shapes.d_model, shapes.ffn

    def randn(*size, scale=1.0):
        return (torch.randn(*size, generator=g, dtype=torch.float64) * scale).to(dtype)

    blocks = [{"fc1.weight": randn(f, d, scale=d ** -0.5), "fc1.bias": randn(f, scale=0.1),
               "fc2.weight": randn(d, f, scale=f ** -0.5), "fc2.bias": randn(d, scale=0.1)}
              for _ in range(shapes.blocks)]
    return {"x": randn(s, b, d), "target": randn(s, b, d), "blocks": blocks}


class Block(nn.Module):
    """out = x + fc2(gelu(fc1(x))). Autograd saves, per call: fc1's input x, gelu's input h, fc2's input a."""

    def __init__(self, d: int, ffn: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(x)))


class Blocks(nn.Module):
    """Blocks with GLOBAL names (blk2.fc1.weight on stage 1 is the same tensor name as in the full model)."""

    def __init__(self, ids, shapes: Shapes) -> None:
        super().__init__()
        self.ids = list(ids)
        self.blk = nn.ModuleDict({str(k): Block(shapes.d_model, shapes.ffn) for k in self.ids})

    def forward(self, x):
        for k in self.ids:
            x = self.blk[str(k)](x)
        return x

    def grads(self) -> dict:
        return {f"blk{k}.{n}": p.grad for k in self.ids for n, p in self.blk[str(k)].named_parameters()}


def stage_blocks(shapes: Shapes, stage: int) -> range:
    per = shapes.blocks // shapes.stages
    assert per * shapes.stages == shapes.blocks
    return range(stage * per, (stage + 1) * per)


def build_blocks(problem: dict, ids, shapes: Shapes, device, dtype) -> Blocks:
    """Blocks loaded with the problem's weights; every .grad pre-allocated as zeros (so a memory reading taken
    later does not include the gradient buffers, and both the baseline and the pipeline accumulate into 0)."""
    m = Blocks(ids, shapes).to(device, dtype)
    with torch.no_grad():
        for k in m.ids:
            for n, p in m.blk[str(k)].named_parameters():
                p.copy_(problem["blocks"][k][n])
                p.grad = torch.zeros_like(p)
    return m


def zero_grads(m: nn.Module) -> None:
    for p in m.parameters():
        p.grad.zero_()


def microbatches(t: torch.Tensor, M: int) -> list[torch.Tensor]:
    """Split [seq, batch, d] along the batch dimension into M equal, CONTIGUOUS microbatches."""
    assert t.shape[1] % M == 0, f"effective batch {t.shape[1]} not divisible by M={M}"
    return [c.contiguous() for c in t.chunk(M, dim=1)]


def mb_loss(y_m: torch.Tensor, t_m: torch.Tensor, n_total: int) -> torch.Tensor:
    """This microbatch's share of the FULL-batch mean squared error: sum over its elements / elements in the batch.

    Summing it over the M microbatches gives exactly the full-batch mean, so the accumulated gradients equal the
    full-batch gradients whatever M is (Day 04's n_m / N rule; for equal microbatches it is mean_m / M, which is
    what Megatron does with `output_tensor /= num_microbatches`)."""
    return ((y_m - t_m) ** 2).sum() / n_total


def compare(name: str, got: torch.Tensor, ref: torch.Tensor, rel_tol: float) -> dict:
    """max abs error, error relative to max|ref|, bitwise flag, and ratio <got,ref>/<ref,ref> (1.0 when right)."""
    got64, ref64 = got.detach().double().cpu(), ref.detach().double().cpu()
    if got64.shape != ref64.shape:
        return {"name": name, "ok": False, "shape": list(got64.shape), "ref_shape": list(ref64.shape),
                "max_abs": float("inf"), "rel": float("inf"), "bitwise": False, "ratio": float("nan")}
    diff = (got64 - ref64).abs().max().item()
    scale = ref64.abs().max().item()
    rel = diff / scale if scale > 0 else diff
    denom = (ref64 * ref64).sum().item()
    ratio = (got64 * ref64).sum().item() / denom if denom > 0 else float("nan")
    return {"name": name, "ok": bool(rel <= rel_tol), "shape": list(got64.shape), "max_abs": diff, "rel": rel,
            "bitwise": bool(torch.equal(got.detach().cpu(), ref.detach().cpu())), "ratio": ratio}


def sha256_tensors(tensors) -> str:
    buf = io.BytesIO()
    torch.save(tensors, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Event log: what each rank actually did, in order
# ---------------------------------------------------------------------------------------------------------------

class EventLog:
    """Ordered record of one rank's compute and P2P calls during one schedule run.

    Compute events carry GPU timestamps (CUDA events on the compute stream, ms since t0) and the number of
    microbatches in flight on this stage right after the event (+1 when a forward starts, -1 when a backward ends).
    Timestamps are for reading the order and the gaps; this small model is launch-bound and they are not a benchmark.
    """

    def __init__(self, rank: int, stage: int, device: torch.device) -> None:
        self.rank, self.stage, self.device = rank, stage, device
        self.events: list[dict] = []
        self.in_flight = 0
        self.peak = 0
        self._cuda = device.type == "cuda"
        self._marks: list = []
        if self._cuda:
            self.t0 = torch.cuda.Event(enable_timing=True)
            self.t0.record()
        else:
            self.t0 = time.perf_counter()

    def _mark(self):
        if self._cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def compute(self, op: str, mb: int, fn, *args):
        start = self._mark()
        if op == "F":
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        out = fn(*args)
        end = self._mark()
        if op == "B":
            self.in_flight -= 1
        self.events.append({"type": "compute", "op": op, "mb": mb, "in_flight": self.in_flight})
        self._marks.append((len(self.events) - 1, start, end))
        return out

    def comm(self, call: str, ops: list) -> None:
        """ops: [("send"|"recv", "a"|"g", mb), ...] (a = activation, g = activation gradient)."""
        self.events.append({"type": "comm", "call": call, "ops": [list(o) for o in ops]})

    def finalize(self) -> list[dict]:
        if self._cuda:
            torch.cuda.synchronize(self.device)
            for i, s, e in self._marks:
                self.events[i]["start_ms"] = round(self.t0.elapsed_time(s), 4)
                self.events[i]["end_ms"] = round(self.t0.elapsed_time(e), 4)
        else:
            for i, s, e in self._marks:
                self.events[i]["start_ms"] = round((s - self.t0) * 1e3, 4)
                self.events[i]["end_ms"] = round((e - self.t0) * 1e3, 4)
        for i, ev in enumerate(self.events):
            ev["seq"] = i
            ev["rank"] = self.rank
            ev["stage"] = self.stage
        return self.events

    # summaries used by the checker and printed in the logs
    def order(self) -> str:
        return " ".join(f"{e['op']}{e['mb']}" for e in self.events if e["type"] == "compute")

    def comm_calls(self) -> list[str]:
        return [f"{e['call']}({','.join(o[1] + str(o[2]) for o in e['ops'])})" for e in self.events
                if e["type"] == "comm"]

    def channel(self, direction: str, kind: str) -> list[int]:
        """Microbatch labels of every send (or recv) of activations ('a') or gradients ('g'), in order."""
        return [o[2] for e in self.events if e["type"] == "comm" for o in e["ops"] if o[0] == direction
                and o[1] == kind]


# ---------------------------------------------------------------------------------------------------------------
# P2P helpers (provided). One stage per rank; stage == rank in this two-rank job.
# ---------------------------------------------------------------------------------------------------------------

class P2P:
    """Megatron-style point-to-point helpers between neighbouring stages (Megatron: P2PCommunicator).

    Every helper is a no-op on the side of the pipeline that has no neighbour (recv_forward on the first stage
    returns None, send_backward on the first stage does nothing, and so on), exactly as in Megatron, so the same
    schedule code runs on every stage.

    The `mb` arguments are LABELS for the event log only. Nothing but the tensor goes over the wire: the receiver
    learns which microbatch it got only from the order (channels are FIFO). The checker compares the sender's
    label sequence with the receiver's to catch a schedule that pairs the wrong microbatches.

    Receive buffers are plain tensors (requires_grad=False). Megatron allocates them with requires_grad=True; here
    making the received activation part of autograd is YOUR job in forward_step (GUIDE §2).
    """

    def __init__(self, stage: int, P: int, shape, dtype, device, log: EventLog) -> None:
        self.stage, self.P = stage, P
        self.prev = stage - 1 if stage > 0 else None
        self.next = stage + 1 if stage < P - 1 else None
        self.shape, self.dtype, self.device, self.log = tuple(shape), dtype, device, log

    def _communicate(self, call, *, send_next=None, send_prev=None, recv_prev=False, recv_next=False, labels=()):
        ops, logged = [], []
        buf_prev = buf_next = None
        if send_prev is not None and self.prev is not None:
            ops.append(dist.P2POp(dist.isend, send_prev.detach().contiguous(), self.prev))
        if recv_prev and self.prev is not None:
            buf_prev = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            ops.append(dist.P2POp(dist.irecv, buf_prev, self.prev))
        if send_next is not None and self.next is not None:
            ops.append(dist.P2POp(dist.isend, send_next.detach().contiguous(), self.next))
        if recv_next and self.next is not None:
            buf_next = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            ops.append(dist.P2POp(dist.irecv, buf_next, self.next))
        logged = [lab for lab, keep in labels if keep]
        if ops:
            # Both directions posted together: neither rank can block the other by sending first (GUIDE §2.4).
            for req in dist.batch_isend_irecv(ops):
                req.wait()
            self.log.comm(call, logged)
        return buf_prev, buf_next

    # --- one direction -----------------------------------------------------------------------------------------
    def recv_forward(self, mb: int):
        """Activation of microbatch mb from the previous stage. None on the first stage."""
        buf, _ = self._communicate("recv_forward", recv_prev=True,
                                   labels=[(("recv", "a", mb), self.prev is not None)])
        return buf

    def send_forward(self, output: torch.Tensor, mb: int) -> None:
        """Send this stage's output (data only; its graph stays here) to the next stage. No-op on the last stage."""
        self._communicate("send_forward", send_next=output, labels=[(("send", "a", mb), self.next is not None)])

    def recv_backward(self, mb: int):
        """d loss / d (this stage's output) for microbatch mb, from the next stage. None on the last stage."""
        _, buf = self._communicate("recv_backward", recv_next=True,
                                   labels=[(("recv", "g", mb), self.next is not None)])
        return buf

    def send_backward(self, input_grad, mb: int) -> None:
        """Send d loss / d (this stage's input) to the previous stage. No-op on the first stage."""
        self._communicate("send_backward", send_prev=input_grad,
                          labels=[(("send", "g", mb), self.prev is not None)])

    # --- both directions at once (1F1B steady state) -----------------------------------------------------------
    def send_forward_recv_backward(self, output: torch.Tensor, send_mb: int, recv_mb: int):
        """Send activation send_mb to the next stage AND receive the gradient for recv_mb from it, posted together."""
        _, buf = self._communicate("send_forward_recv_backward", send_next=output, recv_next=True,
                                   labels=[(("send", "a", send_mb), self.next is not None),
                                           (("recv", "g", recv_mb), self.next is not None)])
        return buf

    def send_backward_recv_forward(self, input_grad, send_mb: int, recv_mb: int):
        """Send gradient send_mb to the previous stage AND receive activation recv_mb from it, posted together."""
        buf, _ = self._communicate("send_backward_recv_forward", send_prev=input_grad, recv_prev=True,
                                   labels=[(("send", "g", send_mb), self.prev is not None),
                                           (("recv", "a", recv_mb), self.prev is not None)])
        return buf


@dataclass
class Stage:
    """What one rank knows. Built by the checker; passed to your forward_step / backward_step / run_* functions."""
    stage: int
    P: int
    module: Blocks                         # this stage's blocks
    comm: P2P
    log: EventLog
    inputs: list = field(default_factory=list)    # first stage: the M input microbatches (on this device)
    targets: list = field(default_factory=list)   # last stage: the M target microbatches
    n_total: int = 0                       # elements in the full-batch output (the loss denominator)
    y_out: torch.Tensor | None = None      # last stage: [seq, batch, d] buffer; save_output() writes into it
    losses: list = field(default_factory=list)    # last stage: detached per-microbatch losses, in microbatch order
    impl: object = None                    # the module holding forward_step / backward_step

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.P - 1

    def save_output(self, mb: int, y: torch.Tensor) -> None:
        """Last stage: keep a detached copy of the model output for the parity check (pre-allocated buffer)."""
        b = y.shape[1]
        self.y_out[:, mb * b:(mb + 1) * b].copy_(y.detach())

    def save_loss(self, mb: int, loss: torch.Tensor) -> None:
        self.losses.append((mb, loss.detach().clone()))

    # Logged wrappers: schedules call THESE, and they call impl.forward_step / impl.backward_step.
    def forward_step(self, mb: int, received):
        return self.log.compute("F", mb, self.impl.forward_step, self, mb, received)

    def backward_step(self, mb: int, input_tensor, output, grad_output):
        return self.log.compute("B", mb, self.impl.backward_step, self, mb, input_tensor, output, grad_output)


# ---------------------------------------------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------------------------------------------

def environment() -> dict:
    info = {
        "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(), "hostname": platform.node(),
        "container_id": os.environ.get("CONTAINER_ID", "?"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        try:
            info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:  # noqa: BLE001
            pass
    return info


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def shapes_to_dict(shapes: Shapes) -> dict:
    return asdict(shapes)


def die_not_implemented(err: NotImplementedError) -> None:
    print(f"\nNOT IMPLEMENTED YET: {err}\nFill in the TODOs in my_pp.py (see GUIDE Block 2).", flush=True)
    sys.exit(3)
