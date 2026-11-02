# Day 04: gradient accumulation and `no_sync()`

K microbatches per optimizer step, run through the same model, data and AdamW as Days 01–03. The variants differ only in **when gradients are communicated**: after every microbatch, or only after the last one (`no_sync()` around the others). Day 02's flat sync, Day 03's `MiniDDP` (extended here with its own `no_sync()`) and real DDP all run in one harness, so parity, collective counts and timings are directly comparable.

Walkthrough: [`day04/GUIDE.md`](../../day04/GUIDE.md). Write-up: [`day04/notes.md`](../../day04/notes.md).

Run once and download before releasing the server (中文): [`day04/RUN_ONCE.zh-CN.md`](../../day04/RUN_ONCE.zh-CN.md). The accompanying `run_server_once.sh` records logs, snapshots the source, runs the full default matrix, checks outputs, and exports a checksummed archive.

## Experiment host

Same as Days 02–03: one Vast.ai instance with **2 × A100-SXM4-40GB**, NV12, torch 2.14.1+cu130, NCCL 2.30.7; `--condition shared`. Today's code imports Day 01's `train.py`, Day 02's `ddp_manual.py` / `check_parity.py` / `launch_utils.py` / `find_symbols.py`, and Day 03's `mini_ddp.py`, all unchanged. The remote project is an upload, so copy this folder next to the others first (use the port and host of your current rental):

```bash
scp -P <port> -r experiments/day04_grad_accum root@<host>:/workspace/AI-Infra-Learning/experiments/
```

## Files

| File | Purpose |
| --- | --- |
| `accum_semantics.py` | One process, no NCCL. Part 1: batch-size bookkeeping (G = W × K × m, optimizer steps per epoch, short final window). Part 2: accumulated vs big-batch gradient and what each scaling bug does. Part 3: does SGD / AdamW / AdamW + clipping notice a wrongly scaled gradient? |
| `mini_ddp_accum.py` | `MiniDDPAccum(MiniDDP)`: `no_sync()`, a forward that arms the hooks only when syncing, a `finish_grad_sync` that is a no-op after a `no_sync` forward, and hook/collective counters. Parts marked **YOU WRITE** |
| `train_accum.py` | One harness: `--impl none / flat_sync / mini_ddp / torch_ddp`, `--sync nosync / every / bwd_only_nosync`, `--global-batch`, `--micro-batch` (K is derived), `--last-global-batch` (short final window), `--bug` (5 negative controls), `--profile`. Saves `result.pt` + `metrics.json` |
| `compare_accum.py` | Orchestrator, plain `python`: smoke → `--stage parity / count / timing / slowlink / all` |
| `count_collectives.py` | NCCL kernels per optimizer step from a Chrome trace (the only way to count DDP's collectives). Pure Python |
| `find_symbols.py` | `file:line` of `no_sync`, `_pre_forward`, `_post_forward`, and every line that touches the two sync flags, in your install; plus the C++ to open on GitHub |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=$(date -u +%Y%m%dT%H%M%SZ)
python experiments/day04_grad_accum/find_symbols.py
python experiments/day04_grad_accum/accum_semantics.py --device cuda
python experiments/day04_grad_accum/compare_accum.py --stage parity   --condition shared --run-id "a100_${RID}"
python experiments/day04_grad_accum/compare_accum.py --stage count    --condition shared --run-id "a100_${RID}" --no-smoke
python experiments/day04_grad_accum/compare_accum.py --stage timing   --condition shared --run-id "a100_${RID}" --no-smoke --repeats 2
python experiments/day04_grad_accum/compare_accum.py --stage slowlink --condition shared --run-id "a100_${RID}" --no-smoke --repeats 2
```

Approximate durations on the dual A100: parity ~3 min, count ~2 min, timing ~25 min (2 repeats), slowlink ~8 min.

One run by hand (K = 4: global batch 64 = 2 ranks × 4 microbatches × 8 rows):

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day04_grad_accum/train_accum.py \
    --impl torch_ddp --sync nosync --preset medium --global-batch 64 --micro-batch 8 --opt-steps 25 --warmup 5 \
    --no-tensors --tag "try_${RID}"
```

Outputs go to `outputs/day04_grad_accum/` (git-ignored): per run `<stage>_<variant>_k<K>_<rid>/{result.pt,metrics.json}`, profiled runs also `trace_rank{0,1}.json` (open in <https://ui.perfetto.dev>), and `count_summary_<rid>.json`, `timing_summary_<rid>.json`, `slowlink_summary_<rid>.json`.

## Notes

- `none` is a **timing reference, not training**: no communication, so the ranks diverge and the checks are skipped.
- Parity reuses Day 02's `check_parity.compare` (A: rtol 1e-4 / atol 1e-6, B 1e-5, C 1e-5) and adds **D** (LR at every optimizer step and Adam's step counter equal the reference, exactly) and **E** (gradient of the short final window, A's tolerance).
- Every microbatch's loss is weighted by `rows_j / rows_in_window` (1/K for equal microbatches). In a short final window that is not 1/K; the `fixed_k_scale` control shows what happens otherwise.
- Timing counts **optimizer steps**: `ms / opt step` covers all K forward/backward passes, the syncs and the optimizer, with `torch.cuda.synchronize()` only at the two ends; slowest rank per step, median over full windows after warmup.
- Timing runs pass `--no-tensors` (metrics only). `/workspace` is a 60 GB container disk.
