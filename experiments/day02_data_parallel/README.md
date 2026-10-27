# Day 02: Data parallel without DDP

N processes (one per GPU) train the Day 01 model on shards of one global batch. Parameters are broadcast once, gradients are averaged with `all_reduce` after every backward, and every rank takes the same optimizer step. No `DistributedDataParallel`. Backend: NCCL on GPUs (`--device cuda`), gloo for a CPU logic check (`--device cpu`).

Walkthrough: [`day02/GUIDE.md`](../../day02/GUIDE.md). Write-up: [`day02/notes.md`](../../day02/notes.md).

## Experiment host: dual A100 (validated 2026-10-27)

Use the single Vast.ai instance with **2 × A100-SXM4-40GB**, topology NV12 and completed default-path NCCL P2P/CUMEM/read. Project: `/workspace/AI-Infra-Learning`; activate project `.venv` (symlink to `/venv/main`). PyTorch 2.14.1+cu130, CUDA build 13.0, NCCL 2.30.7, driver 595.71.05. Requirements allow torch>=2.13.

GPU allocation is exclusive, but the host CPU/resources are shared: use `--condition shared` for consistency with the acceptance evidence. Scale over **1 and 2 GPUs**. Do not disable P2P in normal sessions. Current code passed smoke, ring self-test, tiny parity and all negative controls; medium flat/builtin scaling was measured with 100 steps and three repeats. Full results are in the guide and notes.

## Files

| File | Purpose |
| --- | --- |
| `launch_utils.py` | One env recipe (explicit `OMP_NUM_THREADS`), launching with timeout that only signals the process group it started, smoke test runner |
| `gpu_probe.py` | GPU load, PCIe link (current and max), topology, reported peer-access capability; suggests idle GPUs (`--want N`) |
| `hello_dist.py` | **Smoke test** of torchrun + NCCL/gloo with a timeout; probes `ReduceOp.AVG`. Run first |
| `ddp_manual.py` | Data-parallel training loop (reuses Day 01's model, data, optimizer). Flags: `--sync {per_param,flat}`, `--allreduce {builtin,ring}`, `--bug {none,no_broadcast,no_allreduce,no_average}`, `--global-batch` / `--per-gpu-batch`, `--device {cpu,cuda}`, `--condition {exclusive,shared,unknown}`, `--overwrite` |
| `ring_allreduce.py` | Ring all-reduce from `P2POp` + `batch_isend_irecv`, with a self-test (run under 2 GPU processes; optional odd-world CPU logic tests) |
| `simulate_ring.py` | Pure-Python ring simulator: counts how often each rank's data is summed into each chunk (coverage, not values) |
| `check_parity.py` | 1 GPU vs N GPUs: gradient, 1-update parameters, loss; `--with-bugs` adds negative controls; `--run-id`, `--skip-run`, `--global-batch`, `--omp-threads` |
| `bench_allreduce.py` | All-reduce time and bus bandwidth vs message size (NCCL, optionally the Python ring). Time = max over ranks of per-rank medians |
| `run_scaling.py` | Weak / strong scaling table over 1 and 2 GPUs on this host with `--repeats`, per-repeat baseline, median and min-max |
| `find_symbols.py` | `file:line` of the source-reading symbols in your installed torch; a text-search aid, not proof of a call chain |

## Usage (from the A100 project root)

```bash
cd /workspace/AI-Infra-Learning
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=$(date -u +%Y%m%dT%H%M%SZ)
python experiments/day02_data_parallel/gpu_probe.py --want 2
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/hello_dist.py --device cuda   # smoke test: stop if it fails
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/ring_allreduce.py --device cuda
python experiments/day02_data_parallel/simulate_ring.py 4
python experiments/day02_data_parallel/check_parity.py --device cuda --with-bugs --condition shared
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/bench_allreduce.py --ring --max-mib 256 --warmup 10 --iters 30 --condition shared --tag "a100_bench_${RID}"
python experiments/day02_data_parallel/run_scaling.py --mode weak --preset medium --per-gpu-batch 16 --nprocs 1 2 --steps 100 --repeats 3 --condition shared --run-id "a100_weak_${RID}"
python experiments/day02_data_parallel/run_scaling.py --mode strong --preset medium --global-batch 16 --nprocs 1 2 --steps 100 --repeats 3 --condition shared --run-id "a100_strong_${RID}"
python experiments/day02_data_parallel/find_symbols.py
```

The acceptance upload is a minimal experiment copy, not a full git checkout. Save important results before destroying/recycling this container; `/workspace` has no persistent volume on this rental.

`check_parity.py` and `run_scaling.py` run the smoke test before any long run and stop if it fails. Negative controls are *supposed* to fail check A; the harness prints them as `DETECTED` (good). A control that A misses, or a real variant that fails, is a real failure.

Outputs go to `outputs/day02_data_parallel/<tag>_<run-id>/` (`result.pt`, `metrics.json`; bandwidth sweeps as `<tag>.json`). `result.pt` is a comparison artifact (first-step gradients, first-update parameters, losses, metadata), **not** a resumable optimizer checkpoint. `outputs/` is gitignored, so record key numbers in the notes. Every result records versions, the selected GPU, `NCCL_*`/`OMP_*` overrides, timing method, warmup/timed steps and `--condition`.

## Notes

- Collectives need tensors on the backend's device: CUDA for NCCL. gloo send/recv is CPU-only, so with gloo the ring stages CUDA tensors through host memory.
- `ReduceOp.AVG` worked on both gloo and NCCL on torch 2.14.1 (not tested on 2.13.0). `sync_grads` still does SUM then divide: explicit normalisation that does not depend on backend/version support.
- A peer-access query (`cudaDeviceCanAccessPeer`) reports capability, not working P2P. Check the `NCCL_DEBUG=INFO` transport and run the smoke test; on this host the query, topology, completed smoke and bandwidth comparison together validate the default path.
- The ranks start from different random weights on purpose; rank 0's are broadcast. `--bug no_broadcast` shows why.
- Tolerances in `check_parity.py` are unchanged (A: rtol 1e-4 / atol 1e-6, B: 1e-5, C: 1e-5); observed errors were far below them (see the guide's Appendix A). Final-parameter drift is informational.
- Timing: each step's phases and whole core step are bracketed by `torch.cuda.synchronize()` (CUDA completion is asynchronous to the CPU; current/separate stream selection depends on the version and async_op). Per step the slowest rank counts, then the median over timed steps (core scope). The wider loop scope (data, zeroing, checks, logging) is reported separately. This removes any compute/communication overlap, which Day 3 adds back. Shared-host numbers are provisional.
- Current acceptance results are real A100 measurements dated 2026-10-27. Re-run the smoke and parity when recreating or changing the environment. Do not infer unmeasured training variants from the tiny correctness timings.
