# Day 07: FSDP2 (`fully_shard`), whole model vs per block

This folder runs Day 01's transformer under PyTorch's FSDP2 inside Day 04/06's gradient-accumulation loop, next to DDP with the same AdamW. The FSDP layouts are:

- `root`: one group holding every parameter
- `block`: one group per transformer block, plus the root
- `block_nrf`: per block with `reshard_after_forward=False`
- `block_pf`: per block with explicit forward prefetch

Each layout can reduce-scatter every microbatch (`sync`, FSDP2's default) or only at the window boundary (`nosync`, `set_requires_gradient_sync(False)`). One harness measures parity against one process and DDP, the collectives each layout issues (found in the profiler trace and named by FSDP group), per-rank memory at every block boundary, and step time on NVLink and on a slow link.

Walkthrough: [`day07/GUIDE.md`](../../day07/GUIDE.md). Write-up: [`day07/notes.md`](../../day07/notes.md).

## Experiment host

This runs on the same host as Days 02–06: 2 × A100-SXM4-40GB, NV12, torch 2.14.1+cu130, NCCL 2.30.7, `--condition shared`. The code imports these files unchanged:

- Day 01: `train.py`
- Day 02: `ddp_manual.py`, `check_parity.py`, `launch_utils.py`, `find_symbols.py`, `gpu_probe.py`
- Day 03: `analyze_trace.py` (interval helpers)
- Day 04: `compare_accum.py` (checks D/E), `count_collectives.py` (`kind_of`)

Copy this folder next to the others:

```bash
scp -P <port> -r experiments/day07_fsdp2 root@<host>:/workspace/AI-Infra-Learning/experiments/
```

On a fresh rental, first upload `requirements.txt` and `experiments/day01_single_gpu`, `day02_data_parallel`, `day03_ddp_hooks` and `day04_grad_accum`. Then create the venv link: `ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv`.

## Files

| File | Purpose |
| --- | --- |
| `fsdp_setup.py` | **The part you write.** `broadcast_module_state` (FSDP2 does not broadcast the initial parameters), `apply_fully_shard` (bottom-up wrapping per layout), `set_microbatch_sync` (the accumulation switch); given: explicit forward prefetch, the `sum_grads` control, DTensor → full-tensor evidence helpers, per-rank shard and state reports |
| `train_fsdp.py` | Harness. `--impl single / ddp / fsdp_root / fsdp_block / fsdp_block_nrf / fsdp_block_pf`, `--accum sync / nosync`, `--foreach`, three `--bug` controls, synchronized phase timing, per-rank memory ledger, a block-boundary **memory timeline**, `--memory-snapshot` (for pytorch.org/memory_viz), `--profile` |
| `compare_fsdp.py` | Orchestrator, plain `python`: smoke → `--stage parity / count / memory / timing / slowlink / all` |
| `analyze_fsdp_trace.py` | Finds every NCCL kernel in a trace, ties it to the CPU code that launched it (`FSDP::all_gather (blocks.3)`, `FSDP::post_backward_reduce (blocks.3)`), and reports per-step counts, overlap with compute, exposed communication, GPU idle time, and min-over-ranks kernel durations |
| `find_symbols.py` | `file:line` of the Block 3 reading list in your install |
| `run_server_once.sh` | All stages once, logs, source snapshot, output checks, checksummed archive in `outputs/day07_exports/` |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
nvidia-smi topo -m
python experiments/day07_fsdp2/find_symbols.py
python experiments/day07_fsdp2/compare_fsdp.py --stage parity   --condition shared --run-id "$RID"
python experiments/day07_fsdp2/compare_fsdp.py --stage count    --condition shared --run-id "$RID" --no-smoke
python experiments/day07_fsdp2/compare_fsdp.py --stage memory   --condition shared --run-id "$RID" --no-smoke
python experiments/day07_fsdp2/compare_fsdp.py --stage timing   --condition shared --run-id "$RID" --no-smoke --repeats 3
python experiments/day07_fsdp2/compare_fsdp.py --stage slowlink --condition shared --run-id "$RID" --no-smoke --repeats 2
```

Or run everything at once, with logs and an archive: `bash experiments/day07_fsdp2/run_server_once.sh "$RID"`.

To run one variant by hand (K = 4: global batch 64 = 2 ranks × 4 microbatches × 8 rows):

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day07_fsdp2/train_fsdp.py \
    --impl fsdp_block --accum nosync --preset medium --global-batch 64 --micro-batch 8 --opt-steps 25 --warmup 5 \
    --foreach --no-tensors --tag "try_${RID}"
```

Every run prints one `[rank r]` line per rank. It shows the parameter elements stored versus the full count, `exp_avg` / `exp_avg_sq` MiB, and memory at each point: after `build()`, after wrapping, across the first step, between microbatches, and at the peak. `metrics.json` also holds each rank's memory timeline.

To see the trace of one profiled run, with every collective named by its FSDP group:

```bash
python experiments/day07_fsdp2/analyze_fsdp_trace.py outputs/day07_fsdp2/cnt_block_sync_k1_m8_${RID}/trace_rank0.json --sequence
```

Outputs go to `outputs/day07_fsdp2/` (git-ignored). Each run writes `<stage>_<variant>[_k<K>_m<m>][_r<rep>]_<rid>/{result.pt,metrics.json}`. Profiled runs also write `trace_rank{0,1}.json`, and memory runs write `memory_snapshot_rank0.pickle`. Each stage writes a summary: `count_summary_<rid>.json`, `memory_summary_<rid>.json`, `timing_summary_<rid>.json`, `slowlink_summary_<rid>.json`.

## Notes

- **Matched comparison.** Every timing run uses `AdamW(foreach=True)` for both DDP and FSDP, because Day 01's per-tensor AdamW on DTensors pays DTensor's Python dispatch per op (measured separately: the `per-t` rows). Parity uses Day 01's per-tensor AdamW so it can compare against the 1-process reference, plus one foreach pair (`ddp_foreach` vs `block_nosync_foreach`).
- **R has nothing to check under FSDP.** There are no replicas: every parameter element lives on exactly one rank. R still checks DDP and any parameter FSDP did not manage.
- **FSDP2 does not broadcast the initial parameters.** Ranks are seeded differently on purpose; the harness broadcasts before `fully_shard` (`--bug no_init_bcast` shows what happens otherwise).
- **Kernel durations include waiting.** An NCCL kernel on the rank that arrives first includes waiting for the other rank. The `min-rank` column takes the shorter of the two ranks' durations for the same collective.
- **The memory timeline reads host-side allocator counters** inside module hooks. It needs no synchronization and runs only during microbatch 0 of the last warmup step, which is never timed.
- Timing and count runs save metrics only (`--no-tensors`). `/workspace` is a 60 GB container disk; one full run, traces included, uses about 1 GB.
