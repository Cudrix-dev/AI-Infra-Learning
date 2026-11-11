# Day 09: mixed precision and activation checkpointing

Day 01's medium transformer (Ψ = 27,443,200 parameters, P = 104.69 MiB in FP32) inside Day 04's accumulation window, trained in **fp32**, **bf16 autocast** and **fp16 autocast + GradScaler**, with every block's activations **kept** or **checkpointed**. One GPU answers the core questions; one optional two-GPU stage answers a single distributed question (the dtype DDP all-reduces under autocast).

Walkthrough: [`day09/GUIDE.md`](../../day09/GUIDE.md). Write-up template: [`day09/notes.md`](../../day09/notes.md).

## Experiment host

Same host type as Days 02–08: 2 × A100-SXM4-40GB, NV12, torch 2.14.1+cu130, NCCL 2.30.7, `--condition shared`. No new package. The code imports these files unchanged:

- Day 01: `train.py` (`Config`, `PRESETS`, `TinyTransformer`, `build`, `get_batch`, `setup_reproducibility`, `env_info`)
- Day 02: `ddp_manual.py` (`log0`, `run_metadata`), `launch_utils.py`, `find_symbols.py` (`locate`), `gpu_probe.py`

On a fresh rental, upload `requirements.txt`, `README.md`, `experiments/day01_single_gpu`, `experiments/day02_data_parallel` and this folder, then `ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv`.

## Files

| File | Purpose |
| --- | --- |
| `amp_setup.py` | Reference implementation. **YOU WRITE (1)** `window_step`: one optimizer step over K microbatches with Day 04's weights, clipping and, for fp16, the GradScaler order, plus the `clip_before_unscale` control. **YOU WRITE (2)** `model_forward`: Day 01's forward with `checkpoint(block, ..., use_reentrant=False)`. Given: `autocast_ctx`, `make_scaler`, `sdpa_context`, `AmpNet` |
| `my_amp.py` | The same two functions with empty bodies: write yours here, then pass `--impl mine` |
| `probe_amp.py` | One GPU, one step per configuration. `--part dtypes`: dtype table (params, selected activations, loss, grads, AdamW state, scale), forward ops by dtype, autocast's weight copies, saved-tensor census, gradient error vs fp32, and the "unchanged by step 1" test including `bf16_pure` (no fp32 master). `--part ckpt`: what is saved and recomputed with checkpointing off/on, memory, and bitwise correctness |
| `fp16_order.py` | The debugging exercise: fp32 reference vs fp16 in the right order vs `clip_before_unscale`, checks C0–C3. `--underflow` adds fp16 without a scaler |
| `train_amp.py` | Training harness: `--precision`, `--ckpt`, `--global-batch` / `--micro-batch`, `--max-norm`, `--sdpa`, `--impl`, `--bug`; synchronized per-phase timing and per-phase memory peaks; DDP via `torchrun` with `--comm-hook bf16` and `--profile` |
| `compare_amp.py` | Orchestrator: `--stage matrix` (precision × checkpointing at G = 32, plus optional rows), `--stage ddp` (optional, 2 GPUs) |
| `find_symbols.py` | `file:line` of the Block 3 reading list in your torch install (+ the C++ files to open on GitHub) |
| `run_server_once.sh` | Every stage once, logs, a torch source snapshot, output checks, checksummed archive in `outputs/day09_exports/` |
| `download_results.ps1` | From Windows: download, check the archive SHA-256, extract, re-check every file against the internal manifest |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
D9=experiments/day09_mixed_precision
python $D9/find_symbols.py
python $D9/probe_amp.py --part dtypes --run-id "$RID"                        # required, ~10 s
python $D9/fp16_order.py --run-id "$RID"                                     # required, ~10 s
python $D9/probe_amp.py --part ckpt --run-id "$RID"                          # required, ~10 s
python $D9/compare_amp.py --stage matrix --condition shared --run-id "$RID" --core   # required, ~5 min
python $D9/compare_amp.py --stage ddp --condition shared --run-id "$RID"     # optional, ~1 min
```

Or everything at once, logged and archived: `bash $D9/run_server_once.sh "$RID" --core` (required stages) or without `--core` (adds the optional matrix rows, `--underflow` and the DDP stage). Then, from Windows PowerShell in the project root:

```powershell
experiments/day09_mixed_precision/download_results.ps1 -Remote root@HOST -Port PORT -RunId RUN_ID
```

Your own code: after filling `my_amp.py`,

```bash
python $D9/fp16_order.py --impl mine            # must print FP16 ORDER: ALL CHECKS PASSED
python $D9/probe_amp.py --part ckpt --impl mine # must print CKPT CHECK: PASSED
```

One run by hand:

```bash
python $D9/train_amp.py --precision bf16 --ckpt on --global-batch 32 --micro-batch 32 --opt-steps 40 --tag try_bf16_on
```

Outputs go to `outputs/day09_amp/` (git-ignored): per run `<tag>/metrics.json` (timing, memory, losses, norms seen by clipping, loss scale, parameter SHA-256), JSON summaries `probe_dtypes_*`, `probe_ckpt_*`, `fp16_order_*`, `matrix_summary_*`, `ddp_summary_*`, and profiled DDP runs' `trace_rank{0,1}.json`.

## Notes

- **FP32 means TF32 off** (Day 01's `set_float32_matmul_precision("highest")`). BF16/FP16 are therefore compared with true FP32 CUDA-core matmuls, not with TF32 tensor cores.
- **Attention uses the MATH SDPA backend** by default, as on Days 01–08. It materializes [B, H, T, T] and, for bf16/fp16 inputs, **upcasts q, k, v to fp32 inside the op**. Autocast therefore does not shrink the attention-probability activations. `--sdpa auto` (flash attention) rows are optional.
- **Timing** synchronizes at every phase boundary (forward, backward, optimizer), so the phase times are exact and sum to the step. Median over steps after warmup, 2 repeats in alternating order, `condition=shared`.
- **Memory peaks are per phase** (reset at each boundary). "Start" includes about 64 MiB of cuBLAS workspaces. "Act held" is the memory autograd keeps from forward to backward, with logits already deleted.
- The matrix deletes its `params_final.pt` files (3 × 105 MiB, used for the drift metric) after summarizing. `/workspace` is a 60 GB container disk.
