"""Day 10 - look inside the compiled kernels: launch geometry, registers, spills, occupancy, and what Inductor emits.

    python experiments/day10_triton_softmax/inspect_kernels.py --impl ref [--run-id RID]

Part 1 (Triton, your kernel): for each row length, the launch configuration (BLOCK_SIZE, num_warps, elements per
    thread) and the compiler's metadata for the kernel that was actually built: registers per thread, spilled
    registers, shared memory. From these it ESTIMATES how many programs fit on one SM at once and therefore how many
    rows the grid needs to fill all SMs in one wave. The estimate uses A100 limits (64 warps, 32 blocks, 65,536
    registers and 164 KiB shared memory per SM, registers allocated per warp in units of 256) and is labelled
    [estimate]: the hardware may schedule differently; nothing here is measured.
Part 1b: registers and spills for every num_warps setting that bench_softmax.py --stage warps times.
Part 2 (PTX, fp32, N = 1000 vs N = 1024, both BLOCK 1024): counts of global loads/stores and their vector width,
    exp2 instructions and barriers, so you can connect each line of the kernel to instructions, and see what an
    unaligned row stride does to the loads.
Part 3 (Inductor): the Triton code torch.compile generates for softmax at a short and a long row, and whether it is
    a single-pass "persistent" reduction (whole row in registers, like ours) or a looped reduction.
With --run-id, PTX/TTGIR and Inductor's code are written to outputs/day10_softmax/inspect_<impl>_<run-id>/.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from bench_softmax import WARPS  # noqa: E402
from day10_common import OUT_DIR, PROJECT, env_info, is_not_implemented, load_impl, write_json  # noqa: E402
from softmax_ref import launch_config  # noqa: E402

NS = [128, 1000, 1024, 1025, 2048, 4096, 8192, 16384, 32768, 65536]
SM80 = dict(max_warps=64, max_blocks=32, regs=65536, smem=164 * 1024, reg_unit=256)


def programs_per_sm(n_regs: int, num_warps: int, shared: int) -> int:
    """[estimate] resident programs (thread blocks) per SM, from the three per-SM limits."""
    regs_per_warp = -(-max(n_regs, 1) * 32 // SM80["reg_unit"]) * SM80["reg_unit"]
    by_regs = SM80["regs"] // (regs_per_warp * num_warps)
    by_warps = SM80["max_warps"] // num_warps
    by_smem = SM80["smem"] // shared if shared else SM80["max_blocks"]
    return max(0, min(SM80["max_blocks"], by_warps, by_regs, by_smem))


def ptx_summary(ptx: str) -> dict:
    def forms(op):  # e.g. "ld.global.v4.b32" -> "v4.b32", "ld.global.b32" -> "v1.b32"
        out = set()
        for ins in re.findall(rf"\b{op}\.global[\w.]*", ptx):
            vec = re.search(r"\.(v\d)\b", ins)
            typ = re.search(r"\.([bfu]\d+)$", ins)
            out.add(f"{vec.group(1) if vec else 'v1'}.{typ.group(1) if typ else '?'}")
        return sorted(out)

    return {
        "ld.global": len(re.findall(r"\bld\.global", ptx)),
        "ld.global vector forms": forms("ld"),
        "st.global": len(re.findall(r"\bst\.global", ptx)),
        "st.global vector forms": forms("st"),
        "ex2.approx": len(re.findall(r"\bex2\.approx", ptx)),
        "bar.sync/barrier": len(re.findall(r"\b(?:bar\.sync|barrier\.sync)", ptx)),
        "shfl.sync": len(re.findall(r"\bshfl\.sync", ptx)),
        "div": len(re.findall(r"\bdiv\.(?:rn|full|approx)", ptx)),
        "local (spill) ld/st": len(re.findall(r"\b(?:ld|st)\.local", ptx)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-inductor", action="store_true", help="skip part 3")
    args = ap.parse_args()
    impl = load_impl(args.impl)
    env = env_info()
    sms = env["sm_count"]
    dump = OUT_DIR / f"inspect_{args.impl}_{args.run_id}" if args.run_id else None
    if dump:
        dump.mkdir(parents=True, exist_ok=False)
    print(f"impl={args.impl} | {env['gpu']} ({sms} SMs) | torch {env['torch']} triton {env['triton']}")

    print("\n== Part 1: launch geometry and compiler metadata (one program per row) ==")
    print(f"{'dtype':<5} {'N':>6} {'BLOCK':>6} {'warps':>5} {'thr':>4} {'el/thr':>6} {'regs':>4} {'spill':>5} "
          f"{'smem B':>6} {'prog/SM*':>8} {'rows/wave*':>10}")
    rows, warp_rows, ptx = [], [], {}
    try:
        for dname, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            for n in NS:
                x = torch.randn(4, n, device="cuda", dtype=dtype)
                k = impl.compiled_softmax_kernel(x)
                cfg = launch_config(n)
                n_regs, n_spills = getattr(k, "n_regs", -1), getattr(k, "n_spills", -1)
                shared = int(getattr(k.metadata, "shared", 0))
                pps = programs_per_sm(n_regs, cfg.num_warps, shared)
                r = dict(dtype=dname, n=n, block=cfg.block_size, num_warps=cfg.num_warps,
                         threads=32 * cfg.num_warps, elems_per_thread=cfg.elems_per_thread, n_regs=n_regs,
                         n_spills=n_spills, shared_bytes=shared, est_programs_per_sm=pps, est_rows_per_wave=pps * sms)
                rows.append(r)
                print(f"{dname:<5} {n:>6} {cfg.block_size:>6} {cfg.num_warps:>5} {32 * cfg.num_warps:>4} "
                      f"{cfg.elems_per_thread:>6.1f} {n_regs:>4} {n_spills:>5} {shared:>6} {pps:>8} {pps * sms:>10}")
                if dname == "fp32" and n in (1000, 1024):
                    ptx[n] = k.asm.get("ptx", "") if hasattr(k, "asm") else ""
                    if dump:
                        for key in ("ttir", "ttgir", "llir", "ptx"):
                            if key in getattr(k, "asm", {}):
                                (dump / f"softmax_N{n}_fp32.{key}").write_text(k.asm[key])
        print("* [estimate] from registers/warps/shared memory and A100 per-SM limits; rows/wave = prog/SM x SMs.")
        print("  A grid with fewer rows than rows/wave leaves SM slots idle; regs = registers per thread.")

        print("\n== Part 1b: the num_warps settings timed by bench_softmax.py --stage warps (fp32) ==")
        print(f"{'N':>6} {'warps':>5} {'el/thr':>6} {'regs':>4} {'spill':>5} {'prog/SM*':>8}")
        for n, warps in WARPS.items():
            x = torch.randn(4, n, device="cuda")
            for w in warps:
                k = impl.compiled_softmax_kernel(x, num_warps=w)
                cfg = launch_config(n, w)
                n_regs, n_spills = getattr(k, "n_regs", -1), getattr(k, "n_spills", -1)
                pps = programs_per_sm(n_regs, w, int(getattr(k.metadata, "shared", 0)))
                warp_rows.append(dict(n=n, num_warps=w, elems_per_thread=cfg.elems_per_thread, n_regs=n_regs,
                                      n_spills=n_spills, est_programs_per_sm=pps))
                print(f"{n:>6} {w:>5} {cfg.elems_per_thread:>6.1f} {n_regs:>4} {n_spills:>5} {pps:>8}")
    except Exception as e:  # noqa: BLE001
        if is_not_implemented(e):
            print("\nimpl=mine is not implemented yet: fill in my_softmax.py first.")
            return 3
        raise

    summary = {}
    for n, text in ptx.items():
        print(f"\n== Part 2: PTX of the fp32 N={n} kernel (BLOCK 1024, rows {n} elements = {4 * n} bytes apart) ==")
        summary[n] = ptx_summary(text)
        for key, val in summary[n].items():
            print(f"  {key:<24} {val}")
    print("  (static instruction counts, not executions; v4.b32 = one 16-byte access per thread per instruction.)")
    print("  Triton specializes integer arguments that are multiples of 16 (and 16-byte-aligned pointers): only then "
          "can it\n  prove every row start is 16-byte aligned and emit vector loads/stores.")

    inductor = {}
    if not args.no_inductor:
        print("\n== Part 3: what torch.compile (Inductor) generates for softmax ==")
        from torch._inductor.utils import run_and_get_code

        for m, n in ((4096, 1000), (256, 32768)):
            torch._dynamo.reset()
            x = torch.randn(m, n, device="cuda")
            fn = torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)
            _, codes = run_and_get_code(fn, x)
            code = "\n".join(codes)
            # @triton_heuristics.persistent_reduction: whole row in one block, like ours.
            # @triton_heuristics.reduction: loops over the row in R0_BLOCK chunks (`for r0_offset in tl.range(...)`).
            kind = ("persistent_reduction" if "triton_heuristics.persistent_reduction" in code else
                    "looped reduction" if "triton_heuristics.reduction" in code else "other")
            names = re.findall(r"^def (triton_\w+)", code, flags=re.M)
            loops = len(re.findall(r"for r0?_offset in (?:tl\.)?range", code))
            loads = len(re.findall(r"tl\.load\(in_ptr0", code))
            inductor[f"{m}x{n}"] = dict(kind=kind, kernels=names, row_loops=loops, input_load_sites=loads)
            print(f"  [{m}, {n}] fp32: kernels {names}\n      heuristic = {kind}, {loops} loop(s) over the row, "
                  f"{loads} load site(s) of the input")
            if dump:
                (dump / f"inductor_softmax_{m}x{n}.py").write_text(code)
        print("  'online' in a kernel name = online softmax (running max and rescaled sum in one pass). A looped kernel "
              "with two loops reads the input twice\n  (pass 1: max and sum, pass 2: normalize and store). "
              "Open the dumped code to check.")

    if dump:
        write_json(dump / "inspect.json", dict(env=env, impl=args.impl, kernels=rows, warps=warp_rows,
                                               ptx_fp32={str(k): v for k, v in summary.items()}, inductor=inductor))
        print(f"dumped kernels to {dump.relative_to(PROJECT).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
