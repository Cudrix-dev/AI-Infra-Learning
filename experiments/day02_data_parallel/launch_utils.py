"""Day 02 - shared helpers for check_parity.py and run_scaling.py.

* one environment recipe (explicit OMP_NUM_THREADS for the reference AND the workers)
* subprocess launching that can time out and only ever signals the process group it started
  (never `pkill python`: on a shared server that could hit other people's jobs)
* a smoke test (hello_dist.py) that must pass before any long run starts
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DDP = HERE / "ddp_manual.py"
HELLO = HERE / "hello_dist.py"
OUT = HERE.parents[1] / "outputs" / "day02_data_parallel"

# variables torchrun sets for its workers; remove them so a "single process" run really is single-process
DIST_VARS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
             "MASTER_ADDR", "MASTER_PORT")


def make_env(omp_threads: int, **extra: str) -> dict[str, str]:
    """Inherited environment minus torchrun leftovers, with an EXPLICIT OMP_NUM_THREADS.

    torchrun only sets OMP_NUM_THREADS=1 for its workers when the variable is unset, so without this the
    reference process and the workers could silently use different CPU thread counts.
    NCCL_* variables you exported (e.g. NCCL_P2P_DISABLE=1) are inherited on purpose.
    """
    env = {k: v for k, v in os.environ.items() if k not in DIST_VARS}
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def torchrun_cmd(nproc: int, script: Path, script_args: list[str]) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", str(script), *script_args]


def _descendants(pid: int) -> set[int]:
    """All live descendants of `pid` (children, grandchildren, ...), whatever process group or session they are in.

    torchrun starts each worker in its OWN process group, so signalling the launcher's group never reaches them.
    Must be called BEFORE the launcher dies: afterwards its children are re-parented to init and the link is lost.
    """
    kids: dict[int, list[int]] = {}
    if os.name == "posix" and os.path.isdir("/proc"):
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    else:
        try:
            import psutil  # optional, used on platforms without /proc

            return {c.pid for c in psutil.Process(pid).children(recursive=True)}
        except Exception:  # noqa: BLE001
            return set()
    out, stack = set(), [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie still answers kill(0); treat it as gone
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def _signal(pids, sig) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _terminate(proc: subprocess.Popen, grace: float = 10.0) -> list[int]:
    """Stop the launcher AND every descendant it started. Returns the PIDs that survived even SIGKILL (normally none).

    Only processes in this launcher's own process tree are signalled, never `pkill python`: on a shared server that
    could hit other people's jobs.
    """
    if os.name != "posix":  # taskkill /T walks the tree on Windows
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        return []
    victims = _descendants(proc.pid) | {proc.pid}
    try:
        os.killpg(proc.pid, signal.SIGTERM)  # the launcher's own group (we started it with start_new_session)
    except (ProcessLookupError, PermissionError):
        pass
    _signal(victims, signal.SIGTERM)  # workers live in other groups: signal them directly
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(_alive(p) for p in victims if p != proc.pid):
        time.sleep(0.2)
        victims |= _descendants(proc.pid)
    victims |= _descendants(proc.pid)
    _signal([p for p in victims if _alive(p)], signal.SIGKILL)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(0.5)
    return [p for p in victims if _alive(p)]


def run(cmd: list[str], env: dict[str, str], timeout: float | None = None, capture: bool = False,
        cleanup_grace: float = 10.0) -> tuple[int, str, bool]:
    """Run `cmd`. Returns (returncode, captured_output, timed_out).

    On timeout the whole process tree is stopped (see _terminate) and the remaining output is collected with an upper
    bound: a worker that somehow survived and still holds the output pipe cannot make this function wait forever.
    """
    proc = subprocess.Popen(cmd, env=env, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None,
                            start_new_session=(os.name == "posix"))
    timed_out, out = False, ""
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = e.output if isinstance(e.output, str) else (e.output or b"").decode(errors="replace")
        left = _terminate(proc, cleanup_grace)
        try:
            tail, _ = proc.communicate(timeout=10)  # bounded final read
            out += tail or ""
        except subprocess.TimeoutExpired as e2:
            tail = e2.output if isinstance(e2.output, str) else (e2.output or b"").decode(errors="replace")
            out += tail
            if proc.stdout:
                proc.stdout.close()
        if left:
            out += f"\n[launch_utils] WARNING: PIDs still alive after SIGKILL, stop them yourself: {sorted(left)}\n"
    return proc.returncode if proc.returncode is not None else -9, out or "", timed_out


def smoke_test(nproc: int, device: str, omp_threads: int, timeout_s: int = 120) -> None:
    """Run hello_dist.py on `nproc` ranks. Abort the whole script if it fails or hangs."""
    print(f"\n[smoke] hello_dist.py, {nproc} ranks, device={device} (must pass before any long run)", flush=True)
    cmd = torchrun_cmd(nproc, HELLO, ["--device", device, "--timeout", str(max(30, timeout_s // 2))])
    rc, out, timed_out = run(cmd, make_env(omp_threads), timeout=timeout_s, capture=True)
    if rc == 0 and not timed_out:
        print("[smoke] ok", flush=True)
        return
    tail = "\n".join(out.strip().splitlines()[-25:])
    raise SystemExit(
        f"\n[smoke] FAILED (returncode {rc}, timed out: {timed_out}). Last output:\n{tail}\n\n"
        "Not starting the real run. Look at the log first. Known case (observed 2026-10-21 on one host with 2 x RTX 6000D): "
        "NCCL logged 'via P2P/CUMEM' and the first all-reduce hit 'Watchdog caught collective operation timeout'; "
        "re-running with NCCL_P2P_DISABLE=1 (host-specific workaround, exported only for that run) completed. "
        "That is an observation about that host, not a general fix. If you launched processes that are still alive, "
        "stop only the PIDs you started."
    )
