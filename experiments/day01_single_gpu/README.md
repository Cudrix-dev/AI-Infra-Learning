# Day 01: Reproducible single-GPU baseline

A decoder-only transformer trained on synthetic tokens on one GPU. Later days use it as the baseline: checkpoint resume is bit-exact, timing uses CUDA events with a warmup excluded, and every MiB of memory is accounted for.

Write-up: [`day01/notes.md`](../../day01/notes.md). Walkthrough: [`day01/GUIDE.md`](../../day01/GUIDE.md).

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Training loop, checkpoint save/resume, timing and peak-memory logging, `--memory-probe` |
| `check_resume.py` | Runs full / crash-at-K / resumed (and optionally a bad resume) and compares their losses and final state |
| `profile_attention.py` | PyTorch Profiler: GPU time per kernel for one medium step, math vs fused attention |
| `train_original_backup.py` | The first, unfinished version of `train.py`, kept for reference |

## What `train.py` does

- **Model:** token + position embeddings → L × pre-norm `nn.TransformerEncoderLayer` with a causal mask → LayerNorm → linear head. No dropout.
- **Data:** each row is the sequence `token[i] = (start + stride·i) mod V`, with random start and stride. Every batch is a pure function of `(seed, step)`, so resuming needs only the step counter.
- **Optimizer:** AdamW with `foreach=False, fused=False`, which runs the readable `_single_tensor_adam` path. LR schedule: linear warmup over 20 steps, then cosine decay to 10%.
- **Determinism (default on):** fixed seed, TF32 off, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the MATH attention backend. `--no-deterministic` lets SDPA choose a fused attention kernel.
- **Checkpoint:** model, optimizer, scheduler, CPU/CUDA RNG states, completed step, config. Written atomically (write to `.tmp`, then `os.replace`) and loaded with `torch.load(..., weights_only=True)`.
- **Timing:** CUDA events on every step, a single synchronize at the end, the first `timing_warmup` steps excluded, and peak memory reset after the warmup.

| Preset | B | T | V | D | Layers | Heads | Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tiny` (default) | 16 | 64 | 128 | 128 | 2 | 4 | 0.44M |
| `medium` | 16 | 256 | 8192 | 512 | 6 | 8 | 27.4M |

## Usage (from the repo root)

```bash
# smoke test
python experiments/day01_single_gpu/train.py --tag smoke --steps 30

# resume parity: deterministic must pass with atol = 0; the bad resume must differ
python experiments/day01_single_gpu/check_resume.py --with-bad-resume
python experiments/day01_single_gpu/check_resume.py --no-deterministic --atol 1e-5

# memory lifecycle, phase by phase
python experiments/day01_single_gpu/train.py --memory-probe --preset medium

# baselines
python experiments/day01_single_gpu/train.py --tag base_tiny   --steps 300
python experiments/day01_single_gpu/train.py --tag base_medium --steps 300 --preset medium
python experiments/day01_single_gpu/train.py --tag fast_medium --steps 300 --preset medium --no-deterministic

# where the GPU time goes (run each mode in its own process)
python experiments/day01_single_gpu/profile_attention.py --mode det
python experiments/day01_single_gpu/profile_attention.py --mode fast
```

Main flags of `train.py`: `--preset {tiny,medium}`, `--tag NAME`, `--steps N`, `--batch-size B`, `--stop-at K`, `--resume PATH`, `--bad-resume`, `--no-deterministic`, `--memory-probe`, `--device cpu` (logic smoke test only). Unknown flags passed to `check_resume.py` are forwarded to `train.py`.

Outputs go to `outputs/day01_single_gpu/<tag>/`: `metrics.json` (config, environment, per-step losses, timing, memory) and checkpoints `ckpt_step*.pt` / `final.pt`. `outputs/` is gitignored, so record the key numbers in the notes.

## Baseline results

RTX 5060 Laptop GPU (8 GB), Windows 11, torch 2.13.0+cu130, FP32 (TF32 off). 300 steps, first 10 excluded from timing.

| Run | Median step | tokens/s | Peak allocated |
| --- | ---: | ---: | ---: |
| tiny, deterministic | 9.49 ms | 101.6k | 91.3 MiB |
| medium, deterministic (math attention) | 133.67 ms | 30.4k | 1644.7 MiB |
| medium, fused attention | 124.57 ms | 32.7k | 1406.0 MiB |

- **Resume:** bit-exact (atol = 0) in deterministic mode. The weights-only bad resume differs by up to 0.65 in loss.
- **Steady-state memory on medium:** 3P + 2 × 32 MiB cuBLAS workspaces ≈ 380 MiB. The peak is reached early in backward, while the activations (≈1008 MiB) and two [B·T, V] gradients (256 MiB) are alive at once.
- **Tiny is launch-bound**, so use medium for any A/B comparison. On this laptop, step-time differences under about 3% are noise.

## Known pitfalls

- **Checkpoints must hold only tensors and plain Python types.** `torch.__version__` is a `TorchVersion` object, which `weights_only=True` refuses to load, so it is stored as `str(...)`.
- **`CUBLAS_WORKSPACE_CONFIG` must be set before the first CUDA matmul.** `train.py` sets it with `setdefault`, so a value set in the shell takes priority.
- **`--resume` requires the same shape and schedule config** (including `total_steps`); a mismatch raises an error.
