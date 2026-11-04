"""Day 06 follow-up - runner: arrival skew vs communication, normal P2P vs NCCL_P2P_DISABLE=1. Plain python.

Blocks, executed in this order, one job at a time (never two GPU jobs at once):
  correctness  tiny, 12 steps: sync_diag.py zero1 x {none, prev_step, pre_collective} + flat_ar pre_collective, each
               with --save-tensors and a replica check after every step; references from the UNMODIFIED
               train_zero1.py (zero1, flat_ar) with the same arguments. Final params and gathered AdamW state must be
               bitwise equal (checked by analyze_comm_sync.py)
  route        comm_bench.py, 2 iterations, NCCL_DEBUG=INFO (INIT,GRAPH,P2P,SHM,NET,ENV,TUNING,COLL) to per-process
               files, normal and P2P disabled. Separate from timing because verbose logging perturbs it
  comm         comm_bench.py, 5 warmup + 20 timed iterations, normal and P2P disabled (communication only)
  timing       medium, unprofiled: {flat_ar, zero1} x {p2p, nop2p} x {none, prev_step, pre_collective}, --repeats
               rounds; the configuration order is rotated by 4 positions each round
  profiled     the same 12 configurations once each with the CPU+CUDA profiler (both ranks' traces)
  injected     CONTROL, separately tagged: zero1, normal P2P, rank 1 sleeps --inject-ms after its gradients are ready,
               align {none, prev_step, pre_collective}, one unprofiled and one profiled job each
Every job is a fresh torchrun (fresh NCCL communicator). Normal-P2P workers get NCCL_P2P_DISABLE / NCCL_SHM_DISABLE /
NCCL_ALGO / NCCL_PROTO / NCCL_DEBUG removed; P2P-disabled workers get NCCL_P2P_DISABLE=1. The exact command and
NCCL environment of every job are appended to commands.log before it starts.

  python experiments/day06_comm_sync/run_comm_sync.py --run-id a100_sync_YYYYMMDDTHHMMSSZ
  python experiments/day06_comm_sync/run_comm_sync.py --run-id ... --dry-run      # print the plan and estimate only
Results: outputs/day06_comm_sync/<run-id>/ (see README.md). Then: analyze_comm_sync.py --root that folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from launch_utils import make_env, run, torchrun_cmd  # noqa: E402

DIAG, BENCH, TRAIN0 = HERE / "sync_diag.py", HERE / "comm_bench.py", HERE.parent / "day06_zero1" / "train_zero1.py"
IMPLS, TRANSPORTS, ALIGNS = ["flat_ar", "zero1"], ["p2p", "nop2p"], ["none", "prev_step", "pre_collective"]
SECONDS = {"tiny": 9, "route": 8, "comm": 16, "timing": 11, "profiled": 15, "injected": 13}  # pre-test estimates
STRIP = ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
         "NCCL_DEBUG_FILE")
TINY = ["--preset", "tiny", "--global-batch", "64", "--micro-batch", "8", "--opt-steps", "12"]
MEDIUM = ["--preset", "medium", "--global-batch", "64", "--micro-batch", "8", "--opt-steps", "12"]


def transport_env(transport: str, extra: dict | None = None) -> dict:
    env = {k: v for k, v in make_env(1).items() if k not in STRIP}
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    if transport == "nop2p":
        env["NCCL_P2P_DISABLE"] = "1"
    env.update(extra or {})
    return env


def plan(args, root: Path) -> list[dict]:
    jobs = []
    j = lambda block, tag, cmd, transport="p2p", extra=None: jobs.append(  # noqa: E731
        {"block": block, "tag": tag, "cmd": cmd, "transport": transport, "extra_env": extra or {}})
    jd = root / "jobs"
    # correctness (tiny)
    for impl, align in [("zero1", "none"), ("zero1", "prev_step"), ("zero1", "pre_collective"),
                        ("flat_ar", "pre_collective")]:
        tag = f"tiny_{impl}_{align}"
        j("tiny", tag, torchrun_cmd(2, DIAG, ["--impl", impl, "--align", align, *TINY, "--save-tensors",
                                              "--check-every-step", "--out", str(jd / tag)]))
    for impl in ("zero1", "flat_ar"):
        tag = f"tiny_ref_train_zero1_{impl}"
        j("tiny", tag, torchrun_cmd(2, TRAIN0, ["--impl", impl, *TINY, "--warmup", "4", "--check-every", "1",
                                                "--condition", "shared", "--output-root", str(jd), "--tag", tag]))
    # NCCL route logs and communication-only benchmark
    for t in TRANSPORTS:
        d = root / "route" / t
        j("route", f"route_{t}", torchrun_cmd(2, BENCH, ["--out", str(d), "--iters", "2", "--warmup", "1"]), t,
          {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,P2P,SHM,NET,ENV,TUNING,COLL",
           "NCCL_DEBUG_FILE": str(d / "nccl.%h.%p.log")})
    for t in TRANSPORTS:
        j("comm", f"comm_{t}", torchrun_cmd(2, BENCH, ["--out", str(root / "comm" / t), "--iters", "20",
                                                       "--warmup", "5"]), t)
    # timing (unprofiled) and profiled
    configs = [(i, t, a) for t in TRANSPORTS for i in IMPLS for a in ALIGNS]
    for rep in range(args.repeats):
        k = (4 * rep) % len(configs)
        for impl, t, a in configs[k:] + configs[:k]:
            tag = f"time_{impl}_{t}_{a}_r{rep}"
            j("timing", tag, torchrun_cmd(2, DIAG, ["--impl", impl, "--align", a, *MEDIUM, "--out",
                                                    str(jd / tag)]), t)
    k = (4 * args.repeats) % len(configs)
    for impl, t, a in configs[k:] + configs[:k]:
        tag = f"prof_{impl}_{t}_{a}"
        j("profiled", tag, torchrun_cmd(2, DIAG, ["--impl", impl, "--align", a, *MEDIUM, "--profile", "--out",
                                                  str(jd / tag)]), t)
    # injected-delay control (never mixed into the performance medians)
    for a in ALIGNS:
        for prof in (False, True):
            tag = f"inject_zero1_p2p_{a}_{'prof' if prof else 'noprof'}"
            j("injected", tag, torchrun_cmd(2, DIAG, ["--impl", "zero1", "--align", a, *MEDIUM, "--inject-delay-ms",
                                                      str(args.inject_ms), *(["--profile"] if prof else []),
                                                      "--out", str(jd / tag)]))
    return [x for x in jobs if not args.blocks or x["block"] in args.blocks]


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout


def record_environment(root: Path, when: str) -> None:
    env_dir = root / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / f"nvidia_smi_{when}.txt").write_text(
        sh("nvidia-smi") + sh("nvidia-smi -q -d CLOCK,PERFORMANCE,UTILIZATION")
        + sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"))
    (env_dir / f"host_{when}.txt").write_text(
        "uptime/load: " + sh("uptime") + "affinity of runner: " + sh("taskset -cp $$")
        + "cgroup cpu.max: " + sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null") + sh("lscpu | head -20"))
    if when == "before":
        (env_dir / "topology.txt").write_text(sh("nvidia-smi topo -m"))
        code = ("import json, platform, torch; print(json.dumps({'python': platform.python_version(), "
                "'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'git': torch.version.git_version, "
                "'nccl_runtime': '.'.join(map(str, torch.cuda.nccl.version())), "
                "'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, indent=2))")
        versions = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout
        (env_dir / "versions.json").write_text(versions)
        (env_dir / "hostname.txt").write_text(sh("hostname") + sh("nvidia-smi -L"))


def snapshot_source(root: Path) -> None:
    src = root / "source"
    for rel in ["experiments/day06_comm_sync", "experiments/day06_zero1"]:
        shutil.copytree(ROOT / rel, src / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    for rel in ["experiments/day01_single_gpu/train.py", "experiments/day02_data_parallel/ddp_manual.py",
                "experiments/day02_data_parallel/launch_utils.py", "experiments/day02_data_parallel/ring_allreduce.py"]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, src / rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--inject-ms", type=float, default=10.0)
    ap.add_argument("--blocks", nargs="*", default=None, help="subset of blocks (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = ROOT / "outputs" / "day06_comm_sync" / args.run_id
    jobs = plan(args, root)
    est = sum(SECONDS[x["block"]] for x in jobs)
    counts = {b: sum(x["block"] == b for x in jobs) for b in SECONDS}
    print(f"run id {args.run_id}: {len(jobs)} jobs {counts}; estimated {est / 60:.1f} min", flush=True)
    if args.dry_run:
        for x in jobs:
            print(x["tag"], " ".join(x["cmd"][2:]))
        return 0
    if root.exists():
        raise SystemExit(f"{root} exists: use a new --run-id")
    root.mkdir(parents=True)
    snapshot_source(root)
    record_environment(root, "before")
    sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,clocks.sm,clocks.mem,"
                                "power.draw,temperature.gpu", "--format=csv", "-lms", "1000"],
                               stdout=(root / "env" / "gpu_sampler.csv").open("w"), stderr=subprocess.DEVNULL)
    t_start, failed, done = time.time(), [], []
    try:
        with (root / "commands.log").open("a") as log:
            for i, x in enumerate(jobs):
                env = transport_env(x["transport"], x["extra_env"])
                nccl_env = {k: v for k, v in env.items() if k.startswith(("NCCL_", "TORCH_NCCL_", "CUDA_VISIBLE",
                                                                           "OMP_"))}
                log.write(json.dumps({"i": i, "tag": x["tag"], "block": x["block"], "transport": x["transport"],
                                      "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "env": nccl_env, "cmd": x["cmd"]}) + "\n")
                log.flush()
                if x["block"] == "route":
                    Path(x["extra_env"]["NCCL_DEBUG_FILE"]).parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                rc, out, timed_out = run(x["cmd"], env, timeout=600, capture=True)
                logdir = root / "logs"
                logdir.mkdir(exist_ok=True)
                (logdir / f"{x['tag']}.log").write_text(out)
                status = "ok" if rc == 0 and not timed_out else f"FAILED rc={rc} timeout={timed_out}"
                print(f"[{i + 1}/{len(jobs)}] {x['tag']}: {status} ({time.time() - t0:.1f} s)", flush=True)
                done.append({"tag": x["tag"], "block": x["block"], "rc": rc, "seconds": time.time() - t0})
                if status != "ok":
                    failed.append(x["tag"])
                    if x["block"] == "tiny":
                        raise SystemExit("correctness job failed: not running the timing matrix")
    finally:
        sampler.terminate()
        record_environment(root, "after")
        (root / "jobs_done.json").write_text(json.dumps({"jobs": done, "failed": failed,
                                                         "minutes": (time.time() - t_start) / 60}, indent=2))
    print(f"finished in {(time.time() - t_start) / 60:.1f} min; failed: {failed or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
