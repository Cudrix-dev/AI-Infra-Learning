# Day 06: ZeRO-1 from scratch

AdamW's state (`exp_avg`, `exp_avg_sq`) sharded across the two data-parallel ranks, inside Day 04's accumulation loop. One harness runs the unsharded baselines (DDP + AdamW, and a flat all-reduce with the same flat AdamW as ZeRO-1), the framework's ZeRO-1 (`torch.distributed.optim.ZeroRedundancyOptimizer`), and the hand-written `FlatZeroAdamW` with two accumulation strategies. Parity, optimizer state per rank, collective counts and timings are therefore directly comparable with each other and with Day 04.

Walkthrough: [`day06/GUIDE.md`](../../day06/GUIDE.md). Write-up: [`day06/notes.md`](../../day06/notes.md).

Chinese server run/download checklist for the current rental: [`day06/RUN_SERVER.zh-CN.md`](../../day06/RUN_SERVER.zh-CN.md). Local download and integrity checks: `python experiments/day06_zero1/download_results.py --run-id <RUN_ID>` (defaults to Windows Downloads and `ssh4.vast.ai:17591`).

## Experiment host

Same as Days 02–04: one Vast.ai instance with **2 × A100-SXM4-40GB**, NV12, torch 2.14.1+cu130, NCCL 2.30.7; `--condition shared`. Today's code imports Day 01's `train.py`, Day 02's `ddp_manual.py` / `check_parity.py` / `launch_utils.py` / `find_symbols.py` / `gpu_probe.py`, and Day 04's `compare_accum.py` (checks D/E) and `count_collectives.py`, all unchanged. Copy this folder next to the others (use the port and host of your current rental):

```bash
scp -P <port> -r experiments/day06_zero1 root@<host>:/workspace/AI-Infra-Learning/experiments/
```

On a fresh rental, upload Days 01–04 first (`experiments/day01_single_gpu`, `day02_data_parallel`, `day03_ddp_hooks`, `day04_grad_accum`, plus `requirements.txt`) and create the venv link: `ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv`.

## Files

| File | Purpose |
| --- | --- |
| `zero1.py` | **The core you write.** `FlatZeroAdamW`: one flat parameter buffer (model parameters become views), a persistent flat gradient buffer (or a shard accumulator for `rs_every`), `reduce_grads` (reduce-scatter in place / all-reduce for the unsharded control), AdamW on this rank's slice only, an in-place all-gather. Evidence helpers gather the gradient and the state. Parts marked **YOU WRITE** |
| `train_zero1.py` | Harness. `--impl single / ddp / torch_zero / torch_zero_bucket / flat_ar / zero1 / zero1_rs_every`; Day 04's batch bookkeeping (`--global-batch`, `--micro-batch`, `--last-global-batch`); four `--bug` controls; synchronized phase timing; a per-rank memory ledger and optimizer-state report; `--profile` |
| `compare_zero1.py` | Orchestrator, plain `python`: smoke → `--stage parity / count / timing / slowlink / all` |
| `find_symbols.py` | `file:line` of the reading list in your install: `ZeroRedundancyOptimizer`, `_single_tensor_adam`, the two collectives, `hook_with_zero_step`, FSDP2's `set_requires_gradient_sync` |
| `run_server_once.sh` | The whole matrix once: environment and topology log, source snapshot, all stages, output checks, a checksummed archive in `outputs/day06_exports/` |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
nvidia-smi topo -m
python experiments/day06_zero1/find_symbols.py
python experiments/day06_zero1/compare_zero1.py --stage parity   --condition shared --run-id "$RID"
python experiments/day06_zero1/compare_zero1.py --stage count    --condition shared --run-id "$RID" --no-smoke
python experiments/day06_zero1/compare_zero1.py --stage timing   --condition shared --run-id "$RID" --no-smoke --repeats 3
python experiments/day06_zero1/compare_zero1.py --stage slowlink --condition shared --run-id "$RID" --no-smoke --repeats 2
```

Or everything at once, with logs and an archive: `bash experiments/day06_zero1/run_server_once.sh "$RID"`.

Durations in the pre-test on the dual A100: parity 1.6 min (11 tiny runs), count 1.4 min (6 profiled medium runs), timing 5.7 min (27 medium runs), slowlink 1.8 min (8 runs).

One run by hand (K = 4: global batch 64 = 2 ranks × 4 microbatches × 8 rows):

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day06_zero1/train_zero1.py \
    --impl zero1 --preset medium --global-batch 64 --micro-batch 8 --opt-steps 25 --warmup 5 --no-tensors \
    --tag "try_${RID}"
```

Every run prints one `[state rank r]` line per rank: elements owned, `exp_avg` / `exp_avg_sq` MiB, where `step` lives, the allocation across the first optimizer step, and the memory after wrapping, between microbatches and at the peak.

Outputs go to `outputs/day06_zero1/` (git-ignored): per run `<stage>_<variant>[_k<K>_r<rep>]_<rid>/{result.pt,metrics.json}`, profiled runs also `trace_rank{0,1}.json`, plus `count_summary_<rid>.json`, `timing_summary_<rid>.json`, `slowlink_summary_<rid>.json`.

## Notes

- **Matched controls.** `flat_ar` is `FlatZeroAdamW(shard=False)`: the same flat buffers, gradient views and single-tensor AdamW as `zero1`, with an all-reduce and no sharding. Compare `zero1` with `flat_ar` to see what sharding does; compare `flat_ar` with `ddp` to see what flattening does. Comparing `zero1` with `ddp` mixes the two.
- **Checks.** Parity reuses Day 02's A/B/C and Day 04's D/E against one process with one big batch, and adds **R** (replicas bitwise equal after every step) and **S** (AdamW state gathered from all ranks vs the unsharded DDP + AdamW run). The boundary group (`torch_zero`, `torch_zero_bucket`, `flat_ar`, `zero1`) must also be **bitwise** equal to `ddp`.
- **R is not a correctness check for ZeRO.** The all-gather makes every rank hold the owner's slice, right or wrong; two of the four controls keep R green.
- **Evidence gathers** (full gradient, full state) are extra all-gathers outside the timed region, only when tensors are saved (parity). Timing and count runs pass `--no-tensors`.
- **Phase times** use synchronized boundaries. A collective phase is reported twice: max over ranks (includes waiting for the slower rank) and min over ranks (the last-arriving rank: the collective itself).
- Timing runs save metrics only. `/workspace` is a 60 GB container disk.
