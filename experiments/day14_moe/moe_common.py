"""Day 14 - shared pieces: shapes, Day 01's synthetic tokens, routing patterns, load metrics, comparisons, groups.

Everything that must be IDENTICAL between the one-GPU baseline (EP=1) and the four-rank run (EP=4) lives here:
  * MoEConfig / PRESETS: Day 01's small transformer, with each block's MLP replaced by an MoE layer;
  * make_batch(): Day 01's get_batch() (arithmetic sequences, a pure function of (seed, step));
  * init_value(): every parameter is drawn from its OWN generator keyed by its name, in float64 on the CPU, then
    cast. A rank that builds only expert 3 gets exactly the values the one-GPU model has for expert 3;
  * forced_experts(): reproducible routing patterns defined on the GLOBAL token index, so the one-GPU run and the
    four ranks route every token to the same expert without depending on router rounding;
  * load_metrics(): tokens per expert, max/mean, coefficient of variation, entropies;
  * compare(): the same relative-error rule as Day 11 (max|got - ref| / max|ref|, plus the projection ratio).

Layouts (GUIDE section 4): dense DP is the group over which REPLICATED parameters (attention, norms, embeddings,
router) average their gradients. EP is the group whose members hold DIFFERENT experts and exchange tokens. EDP is
the group whose members hold COPIES of the same experts. With TP = PP = CP = ETP = 1 on four ranks:
    EP=4: dense DP = [0,1,2,3], EP = [0,1,2,3], EDP = [0] [1] [2] [3]
    EP=2: dense DP = [0,1,2,3], EP = [0,1] [2,3], EDP = [0,2] [1,3]     (Megatron's expert order tp-ep-dp-pp)
"""

from __future__ import annotations

import os

# cuBLAS reads this when its handle is created; set before torch touches CUDA (Day 01 does the same).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import zlib  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / "outputs" / "day14_moe"
TENSORS = OUT / "tensors"          # baseline .pt files: small, reproducible, NOT archived

# ----------------------------------------------------------------------------------------------- Day 01 harness
_spec = importlib.util.spec_from_file_location("day01_train", PROJECT / "experiments" / "day01_single_gpu" / "train.py")
day01 = importlib.util.module_from_spec(_spec)
sys.modules["day01_train"] = day01   # dataclasses look their module up in sys.modules while it executes
_spec.loader.exec_module(day01)    # defines Config, get_batch, attention_ctx; main() only runs under __main__


@dataclass(frozen=True)
class MoEConfig:
    vocab: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    seq_len: int = 64
    batch: int = 8            # GLOBAL sequences per step; each of the 4 source ranks gets batch / 4
    n_experts: int = 4        # E; experts are equally sized and distinct
    ffn: int = 512            # expert hidden size (4 x d_model, as Day 01's dense MLP)
    aux_coeff: float = 1e-2   # alpha of L_aux (GUIDE 2.3)
    z_coeff: float = 1e-3     # beta of the router z-loss
    seed: int = 1414

    @property
    def tokens(self) -> int:
        return self.batch * self.seq_len


PRESETS = {
    # parity: small enough for 4 CPU processes, uneven enough to exercise every split case
    "small": MoEConfig(),
    # Day 01's "medium" shapes (vocab 8192, d 512, 8 heads, 6 layers, seq 256, 16 sequences) with 4 experts of 2048
    "medium": MoEConfig(vocab=8192, d_model=512, n_heads=8, n_layers=6, seq_len=256, batch=16, ffn=2048),
}

N_SOURCES = 4   # the global batch is always cut into 4 equal source slices (dense DP = 4 in every 4-GPU layout)
DTYPES = {"float32": torch.float32, "float64": torch.float64}
# Pass rule for every compared tensor: max|got - ref| / max|ref| <= REL_TOL (Day 11's rule). FP32 differences come
# only from a different summation order (different batch sizes in a GEMM, the all-reduce of dense gradients).
REL_TOL = {torch.float32: 1e-5, torch.float64: 1e-12}


def set_exact_math(deterministic: bool = True) -> None:
    """FP32 matmuls in full precision (no TF32) and deterministic kernels where PyTorch has them."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_batch(cfg: MoEConfig, step: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Day 01's synthetic arithmetic sequences, on the CPU: (tokens [B, S], labels [B, S])."""
    c = day01.Config(batch_size=cfg.batch, seq_len=cfg.seq_len, vocab_size=cfg.vocab, seed=cfg.seed)
    return day01.get_batch(c, step, torch.device("cpu"))


def init_value(name: str, shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> torch.Tensor:
    """The value of parameter `name` (e.g. 'blocks.1.moe.experts.3.w1'), independent of which other params exist."""
    g = torch.Generator().manual_seed((zlib.crc32(name.encode()) ^ seed) & 0x7FFFFFFF)
    leaf = name.rsplit(".", 1)[-1]
    if "norm" in name or name.startswith("ln") or ".ln" in name:
        base = 1.0 if leaf == "weight" else 0.0
        return (base + 0.05 * torch.randn(shape, generator=g, dtype=torch.float64)).to(dtype)
    if leaf == "bias" and "router" in name:
        return torch.zeros(shape, dtype=dtype)          # an unbiased router at init (route_stats can skew it)
    if leaf.startswith("b") or leaf == "bias":
        return (0.02 * torch.randn(shape, generator=g, dtype=torch.float64)).to(dtype)
    if "emb" in name:
        return torch.randn(shape, generator=g, dtype=torch.float64).to(dtype) * (1.0 if "tok" in name else 0.1)
    fan_in = shape[-1]
    return (torch.randn(shape, generator=g, dtype=torch.float64) * fan_in ** -0.5).to(dtype)


# ----------------------------------------------------------------------------------------------- routing patterns
PATTERNS = ("learned", "balanced", "hot", "ragged", "all_to_one", "presorted")
# ragged: tokens of source s go to experts in the proportions RAGGED[s] (E = 4 only). Chosen so that there are
# zero-size send splits (source 0 sends only to expert 0), zero-size receive splits, and one expert (2) that
# receives NO token at all, so rank 2's expert runs on an empty batch and must still produce zero gradients.
RAGGED = ((1, 0, 0, 0), (2, 1, 0, 0), (0, 1, 0, 3), (1, 0, 0, 1))
HOT_FRACTION = 0.7


def forced_experts(pattern: str, token_ids: torch.Tensor, n_experts: int, n_tokens_global: int) -> torch.Tensor | None:
    """Expert index of each token for a forced pattern (None = 'learned': use the router's argmax).

    token_ids are GLOBAL token indices (sequence-major: t = sequence * seq_len + position), so the pattern does not
    depend on how the batch is split. The source of token t is t // (n_tokens_global / N_SOURCES).
    """
    if pattern == "learned":
        return None
    t = token_ids.long()
    per_source = n_tokens_global // N_SOURCES
    local = t % per_source
    if pattern == "balanced":                       # every source sends T_s / E to every expert
        return t % n_experts
    if pattern == "hot":                            # ~70 % to expert 0, the rest spread over the others
        h = ((t * 2654435761) % (2 ** 32)).double() / 2 ** 32
        return torch.where(h < HOT_FRACTION, torch.zeros_like(t), 1 + t % (n_experts - 1))
    if pattern == "all_to_one":                     # dropless worst case: one expert (the last) gets every token
        return torch.full_like(t, n_experts - 1)
    if pattern == "presorted":                      # each source's tokens are ALREADY grouped by expert, in order
        return (local * n_experts) // per_source
    if pattern == "ragged":
        assert n_experts == 4, "ragged is defined for E = 4"
        src = t // per_source
        out = torch.empty_like(t)
        for s, weights in enumerate(RAGGED):
            cycle = torch.tensor([e for e, w in enumerate(weights) for _ in range(w)], dtype=torch.long,
                                 device=t.device)
            m = src == s
            out[m] = cycle[local[m] % len(cycle)]
        return out
    raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")


def send_matrix(expert_idx: torch.Tensor, n_experts: int, n_sources: int, ep: int) -> torch.Tensor:
    """[source, destination EP rank] token counts for a global expert assignment (pure arithmetic, no comm).

    Expert e lives on EP rank e // (E / EP) (contiguous blocks, as Megatron's local_expert_indices).
    """
    per_source = expert_idx.numel() // n_sources
    e_loc = n_experts // ep
    m = torch.zeros(n_sources, ep, dtype=torch.long)
    for s in range(n_sources):
        dst = expert_idx[s * per_source:(s + 1) * per_source] // e_loc
        m[s] = torch.bincount(dst, minlength=ep)
    return m


# ----------------------------------------------------------------------------------------------- load metrics
def load_metrics(counts: torch.Tensor) -> dict:
    """Balance of one expert-load vector. Perfect balance: max/mean 1, CV 0, normalized load entropy 1."""
    c = counts.double().cpu()
    n = c.numel()
    mean = c.mean().item()
    q = c / c.sum() if c.sum() > 0 else c
    ent = max(0.0, -(q[q > 0] * q[q > 0].log()).sum().item() / math.log(n)) if n > 1 else 1.0
    return {"counts": [int(x) for x in c.tolist()], "max_over_mean": c.max().item() / mean if mean else float("nan"),
            "cv": (c.std(unbiased=False).item() / mean) if mean else float("nan"), "load_entropy": ent,
            "idle_experts": int((c == 0).sum().item())}


def router_entropy(probs: torch.Tensor) -> float:
    """Mean per-token entropy of the router distribution, normalized to [0, 1] (1 = uniform, 0 = one-hot)."""
    p = probs.double().clamp_min(1e-300)
    return (-(p * p.log()).sum(-1).mean() / math.log(probs.shape[-1])).item()


def capacity(n_tokens: int, n_experts: int, capacity_factor: float, topk: int = 1) -> int:
    """C = ceil(capacity_factor * T * k / E): Megatron's get_capacity(num_tokens=T*k, ...)."""
    return math.ceil(capacity_factor * n_tokens * topk / n_experts)


# ----------------------------------------------------------------------------------------------- comparisons
def compare(name: str, got: torch.Tensor | None, ref: torch.Tensor | None, rel_tol: float) -> dict:
    """max abs error, error relative to max|ref|, bitwise flag, and the projection ratio <got,ref>/<ref,ref>.

    The ratio tells a SCALE bug from a mixing bug: 4.000 means "four times too large" (a missing 1/DP), anything
    that is not a clean number means values from the wrong place. A missing gradient (None) counts as zeros.
    """
    if got is None and ref is None:
        return {"name": name, "ok": True, "rel": 0.0, "max_abs": 0.0, "bitwise": True, "ratio": float("nan")}
    if got is None:
        got = torch.zeros_like(ref)
    if ref is None:
        ref = torch.zeros_like(got)
    got64, ref64 = got.detach().double().cpu(), ref.detach().double().cpu()
    if got64.shape != ref64.shape:
        return {"name": name, "ok": False, "shape": list(got64.shape), "ref_shape": list(ref64.shape),
                "max_abs": float("inf"), "rel": float("inf"), "bitwise": False, "ratio": float("nan")}
    diff = (got64 - ref64).abs().max().item() if got64.numel() else 0.0
    scale = ref64.abs().max().item() if ref64.numel() else 0.0
    rel = diff / scale if scale > 0 else diff
    den = (ref64 * ref64).sum().item()
    ratio = (got64 * ref64).sum().item() / den if den > 0 else float("nan")
    return {"name": name, "ok": bool(rel <= rel_tol), "shape": list(got64.shape), "max_abs": diff, "rel": rel,
            "bitwise": bool(torch.equal(got.detach().cpu(), ref.detach().cpu())), "ratio": ratio}


def tensor_digest(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- process groups
@dataclass
class Groups:
    """This rank's place in the dense-DP / EP / EDP layout (TP = PP = CP = ETP = 1)."""
    world: int = 1
    rank: int = 0
    ep: int = 1
    edp: int = 1
    ep_rank: int = 0
    edp_rank: int = 0
    dp_group: object = None      # dense data parallel: every rank (replicated params average over it)
    ep_group: object = None      # ranks holding DIFFERENT experts that exchange tokens
    edp_group: object = None     # ranks holding the SAME experts (replicas)
    dp_ranks: tuple = (0,)
    ep_ranks: tuple = (0,)
    edp_ranks: tuple = (0,)

    @property
    def dp(self) -> int:
        return self.world


def layout_ranks(world: int, ep: int) -> dict:
    """Rank lists for every group, Megatron's expert order tp-ep-dp-pp with tp = pp = 1: rank = ep_rank + EP * edp_rank."""
    assert world % ep == 0
    edp = world // ep
    return {"dp": [list(range(world))],
            "ep": [[k * ep + j for j in range(ep)] for k in range(edp)],
            "edp": [[j + ep * k for k in range(edp)] for j in range(ep)]}


def build_groups(ep: int) -> Groups:
    """Create the EP and EDP groups. EVERY rank calls new_group for EVERY group, in the same order (Day 13 P3)."""
    world, rank = dist.get_world_size(), dist.get_rank()
    lay = layout_ranks(world, ep)
    g = Groups(world=world, rank=rank, ep=ep, edp=world // ep, ep_rank=rank % ep, edp_rank=rank // ep,
               dp_group=dist.group.WORLD, dp_ranks=tuple(range(world)))
    for ranks in lay["ep"]:
        pg = dist.new_group(ranks)
        if rank in ranks:
            g.ep_group, g.ep_ranks = pg, tuple(ranks)
    for ranks in lay["edp"]:
        pg = dist.new_group(ranks)
        if rank in ranks:
            g.edp_group, g.edp_ranks = pg, tuple(ranks)
    return g


# A mismatched collective should end the job within minutes, not after the 10-30 min default.
TIMEOUT = timedelta(minutes=3)


def init_distributed(cpu: bool) -> tuple[torch.device, str]:
    """Join the job started by torchrun (env://) or by spawn_local (file://). NCCL on GPUs, gloo with --cpu."""
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    if cpu or not torch.cuda.is_available():
        dist.init_process_group("gloo", init_method=os.environ.get("DAY14_INIT", "env://"), timeout=TIMEOUT,
                                rank=rank, world_size=int(os.environ["WORLD_SIZE"]))
        return torch.device("cpu"), "gloo"
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev, timeout=TIMEOUT)
    return dev, "nccl"


def _spawn_entry(rank: int, world: int, init: str, fn, argv):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), DAY14_INIT=init)
    fn(argv)


def spawn_local(fn, argv, nproc: int) -> None:
    """Start `nproc` CPU processes that run fn(argv) with gloo and a file:// rendezvous (no torchrun; works on Windows).

    fn must be a module-level function. Child exit codes propagate: a child that calls sys.exit(3) makes the parent
    raise ProcessExitedException, which main() turns back into exit code 3.
    """
    import torch.multiprocessing as mp
    fd, path = tempfile.mkstemp(prefix="day14_rdzv_")
    os.close(fd)
    os.remove(path)
    init = "file:///" + path.replace("\\", "/").lstrip("/")
    try:
        mp.spawn(_spawn_entry, args=(nproc, init, fn, argv), nprocs=nproc)
    except mp.ProcessExitedException as e:
        sys.exit(e.exit_code if e.exit_code in (2, 3) else 1)
    finally:
        if os.path.exists(path):
            os.remove(path)


class CollectiveCounter:
    """Counts all_to_all_single / all_reduce calls per phase by wrapping the torch.distributed attributes.

    Code that calls `dist.all_to_all_single(...)` at call time is counted (the reference, my_moe.py). Calls made while
    `phase` is None (the checker's own gathers) are not counted.
    """

    NAMES = ("all_to_all_single", "all_reduce")

    def __init__(self) -> None:
        self.phase: str | None = None
        self.counts: dict = {}
        self._orig: dict = {}

    def __enter__(self):
        for name in self.NAMES:
            orig = getattr(dist, name)
            self._orig[name] = orig

            def wrapped(*args, fn_name=name, fn=orig, **kwargs):
                if self.phase is not None:
                    key = f"{self.phase}.{fn_name}"
                    self.counts[key] = self.counts.get(key, 0) + 1
                return fn(*args, **kwargs)

            setattr(dist, name, wrapped)
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(dist, name, orig)

    def take(self) -> dict:
        c, self.counts = self.counts, {}
        return c


# ----------------------------------------------------------------------------------------------- misc
def environment() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "platform": platform.platform(), "hostname": platform.node(),
            "container_id": os.environ.get("CONTAINER_ID", "?"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        try:
            info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:  # noqa: BLE001
            pass
    return info


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def config_dict(cfg: MoEConfig) -> dict:
    return asdict(cfg)


def get_preset(name: str, **overrides) -> MoEConfig:
    return replace(PRESETS[name], **{k: v for k, v in overrides.items() if v is not None})


def incomplete(err: BaseException, where: str = "my_moe.py") -> None:
    """The stub convention of Days 9-13: an unfinished YOU WRITE part prints this and exits with code 3."""
    print(f"\nINCOMPLETE (TODOs left): {err}\nFill in the TODOs in {where} (GUIDE Block 2).", flush=True)
    sys.exit(3)


def sha256_bytes(obj) -> str:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()
