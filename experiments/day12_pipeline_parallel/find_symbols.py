"""Day 12 - file:line of the Block 3 reading list in YOUR Megatron-LM checkout (no torch needed).

    python experiments/day12_pipeline_parallel/find_symbols.py /workspace/Megatron-LM

A text search: `def`/`class` lines, then the first line matching a pattern after a given def. A navigation aid, not
proof of a call chain. The GUIDE's line numbers are for core_v0.19.2 (commit 4b4acac9); run this if your checkout
differs. For a Windows checkout of megatron/core only, see GUIDE Block 3.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SCHED = "megatron/core/pipeline_parallel/schedules.py"
P2P = "megatron/core/pipeline_parallel/p2p_communication.py"
DEFS = [
    (SCHED, ["get_forward_backward_func", "deallocate_output_tensor", "custom_backward", "forward_step_calc_loss",
             "forward_step", "backward_step", "forward_backward_no_pipelining", "get_tensor_shapes",
             "forward_backward_pipelining_without_interleaving"]),
    (P2P, ["_batched_p2p_ops", "_p2p_ops", "P2PCommunicator", "_communicate", "recv_forward", "recv_backward",
           "send_forward", "send_backward", "send_forward_recv_backward", "send_backward_recv_forward"]),
    ("megatron/core/parallel_state.py", ["initialize_model_parallel", "get_pipeline_model_parallel_group"]),
    ("megatron/core/model_parallel_config.py", ["ModelParallelConfig"]),
]
# (file, owner def, regex, what) -> first matching line after the owner's def
INSIDE = [
    (SCHED, "forward_backward_pipelining_without_interleaving", r"num_warmup_microbatches = p2p_communicator\.total",
     "warmup count = stages - stage - 1"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"# Run warmup forward passes", "warmup loop"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"input_tensors\.append\(input_tensor\)",
     "warmup: save input/output for the later backward"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"# Before running 1F1B", "first steady recv_forward"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"# Run 1F1B in steady state", "steady loop"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"send_forward_recv_backward\(",
     "steady: send activation + receive gradient"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"input_tensor = input_tensors\.pop\(0\)",
     "steady: pop the OLDEST microbatch (FIFO)"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"send_backward_recv_forward\(",
     "steady: send gradient + receive next activation"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"# Run cooldown backward passes", "cooldown loop"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"finalize_model_grads_func\(",
     "after the schedule: DP/embedding grad reductions"),
    (SCHED, "forward_step_calc_loss", r"output_tensor /= num_microbatches", "loss divided by num_microbatches"),
    (SCHED, "forward_step", r"set_input_tensor\(input_tensor\)", "received activation handed to the model"),
    (SCHED, "backward_step", r"x\.retain_grad\(\)", "keep the input's gradient"),
    (SCHED, "backward_step", r"torch\.autograd\.backward\(output_tensor\[0\]", "backward with grad_tensors"),
    (SCHED, "backward_step", r"input_tensor_grad\.append\(x\.grad\)", "return the input gradient"),
    (SCHED, "deallocate_output_tensor", r"out\.data = torch\.empty", "free the sent output's data, keep grad_fn"),
    (P2P, "_communicate", r"requires_grad=True", "receive buffers created with requires_grad=True"),
    (P2P, "_communicate", r"elif config\.batch_p2p_comm", "batched path selected"),
    (P2P, "_batched_p2p_ops", r"batch_isend_irecv\(ops\)", "all ops of one call posted together"),
    (P2P, "_p2p_ops", r"if group\.rank\(\) % 2 == 0", "unbatched path: even ranks send first, odd ranks receive first"),
]


def find_def(lines, symbol, start=1, toplevel=False):
    # schedules.py has a docstring example `def forward_step(...)` near the top; its real functions are top level.
    indent = "" if toplevel else r"\s*"
    defn = re.compile(rf"^{indent}(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    for i, line in enumerate(lines[start - 1:], start):
        if defn.search(line):
            return i
    return None


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/Megatron-LM")
    if not (root / "megatron/core").is_dir():
        raise SystemExit(f"{root}/megatron/core not found. Usage: find_symbols.py PATH_TO_MEGATRON_LM")
    try:
        desc = subprocess.run(["git", "-C", str(root), "describe", "--tags"], capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        desc = head = "?"
    print(f"Megatron-LM at {root}: {desc or '?'} {head or ''}\n")
    print("| symbol | location |\n| --- | --- |")
    cache = {}
    for rel, symbols in DEFS:
        lines = cache.setdefault(rel, (root / rel).read_text(encoding="utf-8", errors="replace").splitlines())
        for s in symbols:
            i = find_def(lines, s, toplevel=rel == SCHED)
            print(f"| `{s}` | {rel}:{i if i else '(not found)'} |")
    print("\n| inside | location |\n| --- | --- |")
    for rel, owner, pattern, what in INSIDE:
        lines = cache.setdefault(rel, (root / rel).read_text(encoding="utf-8", errors="replace").splitlines())
        start = find_def(lines, owner, toplevel=rel == SCHED) or 1
        hit = next((i for i, line in enumerate(lines[start - 1:], start) if re.search(pattern, line)), None)
        print(f"| {owner}: {what} | {rel}:{hit if hit else '(not found)'} |")


if __name__ == "__main__":
    main()
