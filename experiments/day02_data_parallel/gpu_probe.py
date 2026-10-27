"""Day 02 - what machine am I on, which GPUs are free, and how are they connected?

Run it at the start AND end of every benchmark session, and paste the output into notes.md section 6.
On a shared server it tells you which GPUs look idle, and prints a CUDA_VISIBLE_DEVICES suggestion.
It only READS state (nvidia-smi, torch). It never changes drivers, BIOS, IOMMU or PCI settings.

  python experiments/day02_data_parallel/gpu_probe.py            # all GPUs
  python experiments/day02_data_parallel/gpu_probe.py --want 4   # also suggest 4 idle GPUs
"""

from __future__ import annotations

import argparse
import subprocess


def sh(cmd: list[str]) -> tuple[bool, str]:
    """(ok, text). On failure the text carries stderr or the exception, so a failed query is visible, not blank."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        return False, f"(failed: {e})"
    if p.returncode != 0:
        return False, f"(exit {p.returncode}: {p.stderr.strip() or p.stdout.strip() or 'no output'})"
    return True, p.stdout.strip()


def query(fields: str) -> list[list[str]] | None:
    ok, out = sh(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not ok:
        print(out)
        return None
    return [[x.strip() for x in line.split(",")] for line in out.splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=0, help="suggest this many idle GPUs")
    ap.add_argument("--idle-mem-mib", type=int, default=500, help="used memory below this counts as idle")
    ap.add_argument("--idle-util", type=int, default=10, help="utilization below this percent counts as idle")
    args = ap.parse_args()
    if args.want < 0:
        raise SystemExit("--want must be >= 0")

    print("== nvidia-smi (index, name, mem used/total MiB, util %, PCIe gen x width: current / max) ==")
    cur = query("index,name,memory.used,memory.total,utilization.gpu,pcie.link.gen.current,pcie.link.width.current")
    mx = query("index,pcie.link.gen.max,pcie.link.width.max")
    maxes = {r[0]: r[1:] for r in (mx or []) if len(r) >= 3}
    idle = []
    for f in cur or []:
        if len(f) < 7:
            print(", ".join(f))
            continue
        i, name, used, total, util, gen, width = f
        gmax, wmax = maxes.get(i, ["N/A", "N/A"])
        try:
            is_idle = int(used) < args.idle_mem_mib and int(util) < args.idle_util
        except ValueError:
            is_idle = False
            used, util = used or "N/A", util or "N/A"
        if is_idle:
            idle.append(i)
        print(f"GPU {i}: {name:28s} {used:>6s}/{total:>6s} MiB  util {util:>3s}%  PCIe gen{gen} x{width} (max gen{gmax} x{wmax})  "
              f"{'idle' if is_idle else 'BUSY or unknown'}")
    print("(current < max can be normal power saving on an idle GPU: re-read under load. If it STAYS lower under load, the slot or\n"
          " motherboard layout limits it - that is a fact about the machine to record, not something to fix on a shared host.)")

    print("\n== topology (nvidia-smi topo -m): NV# = NVLink, PIX/PXB = PCIe switch, PHB/NODE/SYS = through host/CPU ==")
    print(sh(["nvidia-smi", "topo", "-m"])[1])

    try:
        import torch

        n = torch.cuda.device_count()
        print(f"\n== torch {torch.__version__}, CUDA {torch.version.cuda}, NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}, "
              f"{n} visible GPU(s) ==")
        if n > 1:
            print("reported peer-access CAPABILITY (cudaDeviceCanAccessPeer): 1 = the driver says direct access is possible, 0 = it says no.")
            print("This is NOT proof that P2P traffic works: on one observed host the query said 1 and the first NCCL all-reduce still hung.")
            print("To find out what really happens run hello_dist.py with NCCL_DEBUG=INFO under a timeout.")
            print("      " + " ".join(f"{j:>3d}" for j in range(n)))
            for a in range(n):
                print(f"GPU {a} " + " ".join(f"{'-' if a == b else int(torch.cuda.can_device_access_peer(a, b)):>3}" for b in range(n)))
    except Exception as e:  # noqa: BLE001
        print(f"\n(torch not usable here: {e})")

    if args.want:
        if len(idle) >= args.want:
            pick = ",".join(idle[: args.want])
            print(f"\nSuggested: export CUDA_VISIBLE_DEVICES={pick}   (idle now; on a shared server it can change any minute)")
        else:
            print(f"\nOnly {len(idle)} idle GPU(s) right now ({','.join(idle) or 'none'}); wanted {args.want}. Wait, or use fewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
