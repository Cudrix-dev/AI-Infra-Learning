# Day 03: DDP and autograd gradient hooks

`MiniDDP` launches bucketed, asynchronous `all_reduce`s from `register_post_accumulate_grad_hook` during backward, the way `DistributedDataParallel`'s C++ `Reducer` does. One harness (`train_ddp.py`) runs Day 02's synchronous flat sync, `MiniDDP` (per-parameter or bucketed, with optional rebuild and `grad_as_view`) and real DDP on the same model, data and optimizer, so parity, timing and profiler traces are directly comparable.

Walkthrough: [`day03/GUIDE.md`](../../day03/GUIDE.md). Write-up: [`day03/notes.md`](../../day03/notes.md).

## Experiment host

Same as Day 02: one Vast.ai instance with **2 × A100-SXM4-40GB**, NV12, default NCCL `P2P/CUMEM/read`, torch 2.14.1+cu130, NCCL 2.30.7; `--condition shared`. Today's code imports Day 01's `train.py` and Day 02's `ddp_manual.py`, `check_parity.py`, `launch_utils.py`, `find_symbols.py` unchanged. The remote project is an upload, so copy this folder there first:

```bash
scp -P 26936 -r experiments/day03_ddp_hooks root@23.127.144.217:/workspace/AI-Infra-Learning/experiments/
```

## Files

| File | Purpose |
| --- | --- |
| `mini_ddp.py` | `assign_buckets` (DDP's packing rule) and `MiniDDP`: broadcast params+buffers, buckets, post-accumulate hooks, in-order async launch, `finish_grad_sync`; options `rebuild`, `grad_as_view`, `comm=False` (timing ablation); negative controls `early_launch`, `no_wait` |
| `train_ddp.py` | `--impl none / flat_sync / mini_ddp / torch_ddp`, `--bucket-mb` (0 = per parameter), `--first-bucket-mb`, `--rebuild`, `--grad-as-view`, `--no-comm`, `--forward-sync-buffers`, `--bug`, `--profile`; saves `result.pt` + `metrics.json` |
| `compare_ddp.py` | Orchestrator, plain `python`: smoke → `--stage parity / timing / profile / ablation / all` |
| `analyze_trace.py` | Chrome-trace analysis: NCCL time, overlap with other GPU work, exposed NCCL, kernel counts; `--svg` timeline. No torch needed |
| `hook_order.py` | One process: hook semantics, registration vs ready order, bucket layouts and in-order waits |
| `sampler_demo.py` | What `DistributedSampler` gives each rank |
| `find_symbols.py` | Today's Python reading list as `file:line` for your install, plus the C++ files |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=$(date -u +%Y%m%dT%H%M%SZ)
python experiments/day03_ddp_hooks/hook_order.py --device cuda
python experiments/day03_ddp_hooks/hook_order.py --preset medium --bucket-mb 25 --first-bucket-mb 1 --device cuda
python experiments/day03_ddp_hooks/compare_ddp.py --stage parity   --condition shared --run-id "a100_${RID}"
python experiments/day03_ddp_hooks/compare_ddp.py --stage timing   --condition shared --run-id "a100_${RID}" --no-smoke
python experiments/day03_ddp_hooks/compare_ddp.py --stage profile  --condition shared --run-id "a100_${RID}" --no-smoke
python experiments/day03_ddp_hooks/compare_ddp.py --stage ablation --condition shared --run-id "a100_${RID}" --no-smoke --repeats 2
python experiments/day03_ddp_hooks/find_symbols.py
```

Outputs go to `outputs/day03_ddp_hooks/` (git-ignored): per run `<stage>_<variant>_<rid>/{result.pt,metrics.json}`, profiled runs also `trace_rank{0,1}.json` (~11 MB each, open in <https://ui.perfetto.dev>), and `timing_summary_<rid>.json`, `ablation_summary_<rid>.json`, `profile_summary_<rid>.json`, `timeline_<rid>.svg`.

## Notes

Optimizer-only CUDA Graph follow-up: see [the experiment guide](../../day03/CUDA_GRAPH.md).
`run_optimizer_graph.py` compares original AdamW, graph-ready eager AdamW and replay,
and writes to a separate `outputs/day03_cuda_graph/<run-id>/` directory.
The regular harness still defaults to `--optimizer-mode original`.

- `none` and `--no-comm` are **timing references, not training**: ranks diverge and the consistency checks are skipped.
- Parity reuses Day 02's `check_parity.compare` (A: rtol 1e-4 / atol 1e-6, B 1e-5, C 1e-5) and adds a bitwise comparison with `flat_sync` on the same two ranks. With W = 2 every element is fl(a+b), so the pre-test found all variants bitwise identical over 20 steps; that is not expected for W ≥ 3.
- `bug_no_wait` is a race. It is reported but not counted: in the pre-test it was detected in one run and missed in the next.
- Timing: core step only, `torch.cuda.synchronize()` at its two ends (phase boundaries would hide the overlap), slowest rank per step, median over steps; overhead = core(variant) − core(`none`) of the same repeat.
- Profiled runs record ~5,000 CPU ops per step and are much slower than unprofiled ones; use them for structure (which stream, how many kernels, when), and the timing/ablation stages for cost.
- torch 2.14 renamed DDP's `broadcast_buffers` to `forward_sync_buffers` (the old name warns).
