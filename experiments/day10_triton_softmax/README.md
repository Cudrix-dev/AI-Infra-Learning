# Day 10: row-wise fused softmax in Triton

One Triton kernel, one program per row: load the row once, compute `max → exp(x − max) → sum → divide` in registers, store it once. It is checked against `torch.softmax` across 33 row lengths (most of them not powers of two), three dtypes and four layouts, then benchmarked against eager `torch.softmax`, an unfused PyTorch composition, `torch.compile`, and a plain copy of the same bytes. **One GPU.** RMSNorm is a stretch exercise; `torch.compile(mode="reduce-overhead")` is an optional launch-stage comparison.

Walkthrough: [`day10/GUIDE.md`](../../day10/GUIDE.md). Write-up template: [`day10/notes.md`](../../day10/notes.md).

## Experiment host

Same host type as Days 02–09 (2 × A100-SXM4-40GB; this day uses **one** of them through `CUDA_VISIBLE_DEVICES`). torch 2.14.1+cu130 ships **Triton 3.8.0**; no new package. This folder is self-contained: it imports no code from earlier days.

On a fresh rental, upload `README.md`, `requirements.txt`, `day10/` and this folder, then `ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv`.

## Files

| File | Purpose |
| --- | --- |
| `softmax_ref.py` | **Reference.** `softmax_kernel` (the Triton kernel), `launch_config` (BLOCK_SIZE and num_warps from N), `prepare_rows` (the input contract: dtypes, layouts, limits), `softmax` (wrapper, grid = one program per row). Stretch reference: `rmsnorm_kernel`, `rmsnorm` |
| `my_softmax.py` | **Yours.** The same wrapper, with the kernel body left as **YOU WRITE**. Until you fill it, a `tl.static_assert` makes every script report "not implemented yet" (exit code 3). RMSNorm stub for the stretch |
| `softmax_controls.py` | Three deliberately broken kernels (`no_max`, `zero_fill`, `no_store_mask`), used by `check_softmax.py --controls` |
| `check_softmax.py` | Correctness: row-length sweep × dtypes × layouts (including a sentinel-padded output that proves masked stores), stability cases, `-inf` masks, contract errors, float64 error report. `--impl ref|mine`, `--op softmax|rmsnorm`, `--controls`, `--quick` |
| `inspect_kernels.py` | Launch geometry, registers, spills and an occupancy **estimate** per row length; PTX instruction counts; the Triton code `torch.compile` generates (persistent vs looped/online reduction) |
| `bench_softmax.py` | Timing: cold first call, warmup, steady state per call (`sync_us`), back-to-back (`loop_us`) and CUDA-Graph replay (`graph_us`); effective GB/s. `--stage quick|rows|count|padding|warps|launch|core|all` |
| `find_symbols.py` | `file:line` of the Block 3 reading list in your Triton and torch install |
| `day10_common.py` | Impl loader, tolerances, environment record, JSON writer |
| `run_server_once.sh` | Every stage once, logged, output-checked, archived with SHA-256 in `outputs/day10_exports/` |
| `download_results.ps1` | From Windows: download, check the archive SHA-256, extract, re-check every file against the internal manifest |

## Contract (what the kernel supports)

| | Supported | Rejected with `ValueError` |
| --- | --- | --- |
| Device / dtype | CUDA; float32, float16, bfloat16 (computed in float32, rounded once on store) | CPU tensors, float64, integer types |
| Layout | last dim contiguous; 2-D with any row stride; N-D viewable as `[rows, N]` (any contiguous tensor) | last dim with stride ≠ 1; N-D that cannot be flattened without a copy |
| Row length | 1 ≤ N ≤ 65,536 (whole row in one program's registers) | N > 65,536 |
| `out=` | 2-D, same shape/dtype, contiguous last dim, any row stride | anything else |
| Semantics | softmax over the last dim; `-inf` entries → 0; an all-`-inf` row → NaN (as torch) | — |

Tolerances against `torch.softmax` in the same dtype (`day10_common.TOLERANCES`): fp32 rtol 1e-5 / atol 1e-6, fp16 rtol 1e-3 / atol 1e-5, bf16 rtol 1.6e-2 / atol 1e-5 (about one ulp of fp16, two of bf16, both sides rounding an fp32 result).

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
D10=experiments/day10_triton_softmax
RID=manual_$(date -u +%Y%m%dT%H%M%SZ)
python $D10/find_symbols.py
python $D10/check_softmax.py --impl ref --controls --run-id "$RID"     # ~20 s; SOFTMAX CHECK: PASSED + CONTROLS: ALL DETECTED
python $D10/inspect_kernels.py --impl ref --run-id "$RID"              # ~20 s
python $D10/bench_softmax.py --impl ref --stage core --run-id "$RID"   # ~70 s
python $D10/bench_softmax.py --impl ref --stage launch --run-id "$RID" # optional, ~25 s
```

Your kernel, after filling `my_softmax.py`:

```bash
python $D10/check_softmax.py --impl mine --quick        # fast loop while writing (13 row lengths)
python $D10/check_softmax.py --impl mine                # must print SOFTMAX CHECK: PASSED
python $D10/bench_softmax.py --impl mine --stage quick  # should match the reference within noise
```

Or everything at once, logged and archived (no automatic shutdown):

```bash
bash $D10/run_server_once.sh "$RID" --impl ref --full    # reference, all stages (~2.5 min)
bash $D10/run_server_once.sh "${RID}_mine" --impl mine   # your kernel, required stages
```

Then, from Windows PowerShell in the project root:

```powershell
experiments/day10_triton_softmax/download_results.ps1 -Remote root@HOST -Port PORT -RunId RUN_ID
```

Outputs go to `outputs/day10_softmax/` (git-ignored): `check_<op>_<impl>_<RID>.json`, `inspect_<impl>_<RID>/` (PTX, TTGIR, Inductor code, `inspect.json`), `bench_<impl>_<stage>_<RID>.{json,csv}`, and `session_<RID>/` with every log, `environment.log`, `nvidia_smi_q.txt`, a project snapshot, the installed Triton/Inductor sources read in Block 3, and `SHA256SUMS`.

## Notes

- **Effective GB/s** = 2·M·N·element_size / time: one read and one write per element, the minimum for any softmax. It is not measured DRAM traffic. The `copy` row gives the achieved bandwidth of a plain copy of the same bytes; the A100-40GB data-sheet peak is 1555 GB/s. Rows flagged `L2` (bytes ≤ 40 MiB) can be served from L2 across back-to-back calls and are not DRAM measurements.
- **Three steady-state timings.** `sync_us`: one call + synchronize (latency of an isolated call). `loop_us`: back-to-back calls between CUDA events (max of host submission and GPU time). `graph_us`: the same calls captured in one CUDA Graph and replayed (≈ GPU time). `loop_us − graph_us` is the host-submission cost visible to the loop.
- **Cold compile.** The runner sets `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` to fresh directories under `/tmp` (deleted at the end), so `first_ms` includes compilation. Running the scripts by hand reuses `~/.triton/cache` and Inductor's cache, so later first calls are faster.
- **Compiled baselines** are rebuilt per shape after `torch._dynamo.reset()` with `dynamic=False`, so every shape gets its own specialized kernel and Dynamo's recompile limit is never hit.
