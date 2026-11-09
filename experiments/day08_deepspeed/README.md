# Day 08: DeepSpeed ZeRO stages 0, 1, 2, 3 on the same model

This folder runs Day 01's transformer under DeepSpeed's engine at ZeRO stage 0, 1, 2 and 3, inside Day 04's gradient-accumulation loop, next to torch DDP with the same AdamW. One harness measures:

- parity against one process and against DDP, with two DeepSpeed-specific negative controls;
- the collectives each stage issues (NCCL kernel names in the profiler trace, per optimizer step, with their phase and overlap);
- per-rank memory at every block boundary, plus a **tensor census** that names every CUDA buffer DeepSpeed allocated;
- step time on NVLink and on a slow link, with buckets sized for this model and with DeepSpeed's defaults.

Walkthrough: [`day08/GUIDE.md`](../../day08/GUIDE.md). Write-up: [`day08/notes.md`](../../day08/notes.md).

## Experiment host and install

Same host type as Days 02–07: 2 × A100-SXM4-40GB, NV12, torch 2.14.1+cu130, NCCL 2.30.7, `--condition shared`. DeepSpeed is new today. Install it once per rental, without building any CUDA extension (the client `torch.optim.AdamW` needs none):

```bash
source .venv/bin/activate
DS_BUILD_OPS=0 python -m pip install deepspeed==0.19.9
python -c "import deepspeed; print(deepspeed.__version__)"
```

The code imports these files unchanged:

- Day 01: `train.py`
- Day 02: `ddp_manual.py`, `check_parity.py`, `launch_utils.py`, `find_symbols.py`, `gpu_probe.py`
- Day 03: `analyze_trace.py` (interval helpers)
- Day 04: `compare_accum.py` (checks D/E), `count_collectives.py` (`kind_of`)
- Day 07: `train_fsdp.py` (`MemTimeline`, `alloc_mib`), `analyze_fsdp_trace.py` (kernel → launch → annotation)

On a fresh rental, upload `requirements.txt` and those five experiment folders plus this one, then `ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv`.

## Files

| File | Purpose |
| --- | --- |
| `ds_setup.py` | **The part you write.** `make_ds_config` (one config per stage; buckets sized for this model or DeepSpeed's defaults) and `ds_microbatch` (Day 04's accumulation contract through `engine.backward` / `set_gradient_accumulation_boundary`). Given: evidence helpers that reassemble full gradients, fp32 parameters and AdamW state from the shards (`deepspeed.utils.safe_get_full_*`), shard and state reports, and `tensor_census` |
| `train_ds.py` | Harness. `--impl single / ddp / ds`, `--zero-stage 0-3`, `--buckets sized / default`, `--overlap auto / on / off`, `--foreach`, two `--bug` controls, synchronized phase timing, per-rank memory ledger, Day 07's block-boundary memory timeline, `--census`, `--memory-snapshot`, `--profile` |
| `compare_ds.py` | Orchestrator, plain `python`: smoke → `--stage parity / count / memory / timing / slowlink / all` |
| `find_symbols.py` | `file:line` of the Block 3 reading list in your DeepSpeed install |
| `run_server_once.sh` | All stages once, logs, DeepSpeed source snapshot, output checks, checksummed archive in `outputs/day08_exports/` |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
nvidia-smi topo -m
python experiments/day08_deepspeed/find_symbols.py
python experiments/day08_deepspeed/compare_ds.py --stage parity   --condition shared --run-id "$RID"
python experiments/day08_deepspeed/compare_ds.py --stage count    --condition shared --run-id "$RID" --no-smoke
python experiments/day08_deepspeed/compare_ds.py --stage memory   --condition shared --run-id "$RID" --no-smoke
python experiments/day08_deepspeed/compare_ds.py --stage timing   --condition shared --run-id "$RID" --no-smoke --repeats 3
python experiments/day08_deepspeed/compare_ds.py --stage slowlink --condition shared --run-id "$RID" --no-smoke --repeats 2
```

Run the three required stages with `bash experiments/day08_deepspeed/run_server_once.sh "$RID" --core`. Omit `--core` to include timing and slow-link runs. Both modes log each stage, verify the required outputs and package a checksummed archive. `CORE_SUCCESS.txt` records that the required checks passed even if an optional stage later fails.

From Windows PowerShell, run `experiments/day08_deepspeed/download_results.ps1 -Remote root@HOST -Port PORT -RunId RUN_ID` to download, verify and extract the archive to `outputs/day08_downloads/RUN_ID/`.

One variant by hand (K = 4: global batch 64 = 2 ranks × 4 microbatches × 8 rows):

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day08_deepspeed/train_ds.py \
    --impl ds --zero-stage 3 --preset medium --global-batch 64 --micro-batch 8 --opt-steps 25 --warmup 5 \
    --foreach --no-tensors --census --tag "try_${RID}"
```

Every run prints one `[rank r]` line per rank: parameter elements kept resident versus the full count (stage 3: the partition plus the "persistent" small parameters), the fp32 master elements, `exp_avg` / `exp_avg_sq` MiB, and memory after `build()`, after `deepspeed.initialize`, across the first step, between microbatches and at the peak. With `--census` rank 0 also prints which buffer owns each MiB at the start of the step and right before the optimizer.

Outputs go to `outputs/day08_deepspeed/` (git-ignored): `<stage>_<variant>[_k<K>_m<m>][_r<rep>]_<rid>/{result.pt,metrics.json}`, traces for profiled runs, `memory_snapshot_rank0.pickle` for memory runs, and one `<stage>_summary_<rid>.json` per stage.

## Variant names

| Name | Flags | What it isolates |
| --- | --- | --- |
| `ddp` | `--impl ddp` | torch DDP, 25 MB buckets: the Day 07 baseline |
| `z0` | `--impl ds --zero-stage 0` | DeepSpeed's engine with nothing sharded |
| `z1`, `z2`, `z3` | `--zero-stage 1/2/3` | buckets sized for this model (25 MiB; stage 3 prefetch and reuse ≈ one block) |
| `z2_ovl` | `--zero-stage 2 --overlap on` | stage 2 with `overlap_comm` (reduction on a side stream, two bucket buffers) |
| `z1_def`, `z2_def`, `z3_def` | `--buckets default` | DeepSpeed's default bucket / prefetch / reuse sizes, tuned for billion-parameter models |

## Notes

- **Matched comparison.** Timing runs use `AdamW(foreach=True)` for every variant. Under ZeRO stages 1–3 the client AdamW steps one flat fp32 partition per param group, so `foreach` changes little there; it matters for `ddp` and `z0` (78 tensors).
- **DeepSpeed rewrites your optimizer.** `deepspeed.initialize` replaces the client AdamW's `param_groups[i]["params"]` with a flat fp32 partition. The LR scheduler built on the client optimizer keeps working because the group dicts are the same objects. The harness steps it itself, once per window.
- **Evidence is collective.** `safe_get_full_grad` / `safe_get_full_fp32_param` / `safe_get_full_optimizer_state` all-reduce a fragment into a full tensor, so every rank calls them in the same order. Gradients are read between the boundary backward and `engine.step()`, because ZeRO frees them inside the step.
- **Kernel names, not DeepSpeed's names.** The count stage classifies NCCL kernels (`AllReduce`, `ReduceScatter`, `AllGather`, `Reduce`). Stage 1/2's "reduce-scatter" shows up as `AllReduce` with DeepSpeed's default `use_multi_rank_bucket_allreduce=True` (guide Block 3, Stop 4).
- Timing and count runs save metrics only (`--no-tensors`). `/workspace` is a 60 GB container disk; a full run with traces uses well under 2 GB.
