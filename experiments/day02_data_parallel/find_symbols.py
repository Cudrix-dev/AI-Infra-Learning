"""Day 02 - print file:line for every symbol on today's source-reading list, in YOUR installed torch.

Line numbers move between releases, so this script looks them up instead of hard-coding them.
Output is a Markdown table you can paste into notes.md section 3.

CAVEAT: this is a TEXT search for "def name" / "class name" lines. It is a navigation aid, not proof of a call chain:
a match can be a different function with the same name, a wrapper, or dead code, and a missing match can just mean the
symbol moved or was renamed in your version. Confirm any call chain by reading the code (or by a debugger / stack trace).

  python experiments/day02_data_parallel/find_symbols.py
  python experiments/day02_data_parallel/find_symbols.py all_reduce      # only symbols containing this text
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

ROOT = Path(torch.__file__).resolve().parent  # .../site-packages/torch

# (file relative to the torch package, [symbols]) in READING ORDER
READING_LIST = [
    ("distributed/run.py", ["get_args_parser", "config_from_args", "run_script_path", "main"]),
    ("distributed/launcher/api.py", ["elastic_launch", "launch_agent"]),
    ("distributed/elastic/agent/server/local_elastic_agent.py", ["_start_workers"]),
    ("distributed/rendezvous.py", ["_env_rendezvous_handler", "_create_c10d_store"]),
    ("distributed/distributed_c10d.py", [
        "init_process_group", "_new_process_group_helper", "_get_default_group", "get_rank", "get_world_size",
        "_rank_not_in_group", "all_reduce", "broadcast", "ReduceOp", "isend", "irecv", "batch_isend_irecv", "P2POp",
        "barrier", "_coalescing_manager"]),
    ("distributed/utils.py", ["_sync_module_states", "_verify_param_shape_across_processes"]),
    ("nn/parallel/distributed.py", ["DistributedDataParallel", "no_sync", "_pre_forward"]),
]


def locate(path: Path, symbol: str) -> str:
    if not path.exists():
        return "(file not found)"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    defn = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b|^{re.escape(symbol)}\s*[:=]")
    for i, line in enumerate(text, 1):
        if defn.search(line):
            return f"{path.relative_to(ROOT.parent).as_posix()}:{i}"
    mention = re.compile(rf"\b{re.escape(symbol)}\b")
    for i, line in enumerate(text, 1):
        if mention.search(line):
            return f"(no def here; first mention {path.relative_to(ROOT.parent).as_posix()}:{i})"
    return "(not found: moved or renamed in this version, try grep -rn)"


def main() -> None:
    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"torch {torch.__version__} at {ROOT}\n")
    print("| Symbol | Where in your install |\n| --- | --- |")
    for rel, symbols in READING_LIST:
        for s in symbols:
            if needle in s or needle in rel:
                print(f"| `{s}` | {locate(ROOT / rel, s)} |")
    print("\nC++ side (not shipped in the wheel; read it on GitHub at the tag matching your version, e.g. v2.13.0):")
    for f, syms in [
        ("torch/csrc/distributed/c10d/ProcessGroupGloo.cpp", "ProcessGroupGloo::allreduce, AsyncAllreduceWork, ProcessGroupGloo::broadcast, ProcessGroupGloo::send/recv"),
        ("torch/csrc/distributed/c10d/Ops.cpp", "allreduce_ (the dispatcher op that Python's all_reduce ends up calling)"),
        ("torch/csrc/distributed/c10d/Backend.hpp", "Backend::allreduce (the virtual function each backend overrides)"),
        ("torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp", "ProcessGroupNCCL::allreduce, the collective<> helper, getNCCLComm, WorkNCCL::wait / synchronizeStream (the GPU path you use today)"),
        ("NVIDIA/nccl: src/collectives.cc, src/enqueue.cc, src/device/all_reduce.h", "ncclAllReduce; ring vs tree algorithms (read it on GitHub; NCCL_DEBUG=INFO prints the rings it built)"),
        ("third_party/gloo/gloo/allreduce.cc", "gloo::allreduce (the CPU path: which algorithm does it pick?)"),
    ]:
        print(f"  - {f}: {syms}")


if __name__ == "__main__":
    main()
