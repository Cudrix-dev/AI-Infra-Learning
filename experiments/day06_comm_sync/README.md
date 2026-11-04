# Day 06 follow-up: arrival skew vs communication, normal P2P vs `NCCL_P2P_DISABLE=1`

A focused diagnostic for one question from Day 06: in a non-overlapped collective after backward, how much of the
measured time is **one rank waiting for the other**, and how much is the collective itself? And does synchronizing the
two processes after the previous iteration (the "prev_step" proposal) remove the waiting from the next iteration's
traces?

Report: [`day06/COMM_SYNC.md`](../../day06/COMM_SYNC.md). The original Day 06 harness in `experiments/day06_zero1/` is
unchanged; this folder imports `zero1.py` (FlatZeroAdamW), Day 01's `train.py` and Day 02's helpers.

## Files

| File | Purpose |
| --- | --- |
| `sync_diag.py` | The `flat_ar` / `zero1` loop of `train_zero1.py` (same math, seeds, schedule, local synchronizations) plus `--align none / prev_step / pre_collective` (CPU Gloo barrier after a local `torch.cuda.synchronize`), `--inject-delay-ms` (rank-1 control), `--profile` (CPU+CUDA, steps 6–8 recorded), raw per-step per-rank host timestamps in `rows_rank{r}.jsonl` and named `record_function` markers |
| `comm_bench.py` | Communication only: all-reduce, reduce-scatter, all-gather and an RS→AG pair on 27,443,200 FP32 elements with FlatZeroAdamW's in-place layouts; barrier-aligned, host + CUDA-event timing, exact output validation |
| `run_comm_sync.py` | The matrix (correctness, NCCL route logs, comm-only, unprofiled timing, paired traces, injected delay), one fresh `torchrun` per job, exact commands and NCCL environment in `commands.log`, environment/topology/clock snapshots, GPU sampler |
| `analyze_comm_sync.py` | Pairs rank 0 and rank 1 per (job, step, operation); pairs NCCL kernels to operations through the launch correlation id; writes `rows.csv`, `tables.md`, `summary.json` |

## Usage (A100 project root)

```bash
RID=a100_sync_$(date -u +%Y%m%dT%H%M%SZ)
python experiments/day06_comm_sync/run_comm_sync.py --run-id "$RID" --dry-run     # plan + job count + estimate
python experiments/day06_comm_sync/run_comm_sync.py --run-id "$RID"               # 64 jobs, ~13 min
python experiments/day06_comm_sync/analyze_comm_sync.py --root "outputs/day06_comm_sync/$RID"
```

One job by hand:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day06_comm_sync/sync_diag.py \
    --impl zero1 --align pre_collective --profile --out outputs/day06_comm_sync/try/jobs/x
NCCL_P2P_DISABLE=1 python -m torch.distributed.run --standalone --nproc_per_node 2 \
    experiments/day06_comm_sync/comm_bench.py --out outputs/day06_comm_sync/try/comm/nop2p
```

## Alignment modes

| Mode | Where the CPU Gloo barrier runs | Tests |
| --- | --- | --- |
| `none` | nowhere (local synchronizations only, as in `train_zero1.py`) | baseline |
| `prev_step` | at the top of every iteration s, after iteration s−1 finished including its `prof.step()` | "synchronize after one iteration, trace the next" |
| `pre_collective` | after the local gradients are GPU-ready, immediately before `reduce_grads()`; for `zero1` also immediately before the parameter all-gather | alignment right at the collective |

Every barrier is preceded by `torch.cuda.synchronize()` of the rank's own device. The Gloo group launches no GPU
kernel and is invisible to FlatZeroAdamW's collective counters. A barrier aligns when the ranks *leave* it, not when
their next kernels start; the residual is measured (`bar_exit_gap`, `arrival_gap`, kernel start gap).

## Notes

- The injected-delay jobs (`inject_*`) are controls, tagged separately and excluded from all performance rows.
- The `comm_bench` RS→AG pair has no division and no optimizer between the collectives: it is not a ZeRO-1 step.
- Event intervals in `comm_bench` are recorded on the current stream around the call. They include the NCCL stream's
  dependency and any residual waiting; they are not isolated transfer times.
- Outputs: `outputs/day06_comm_sync/<run-id>/` (git-ignored): `jobs/<tag>/` (rows, meta, traces, tiny results),
  `comm/`, `route/` (NCCL INFO logs), `logs/`, `env/`, `source/`, `commands.log`, `rows.csv`, `tables.md`,
  `summary.json`, `report.md`, `SHA256SUMS`.
