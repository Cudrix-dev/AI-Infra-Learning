# Day 13: Megatron Core at DP=1, TP=2, PP=2 on four GPUs

A small GPT (Day 01 medium shapes: d 512, 8 heads, 6 layers, vocab 8192, sequence 256, 8 microbatches × 2 sequences, FP32, TF32 off, dropout 0, untied embeddings) trained for 5 steps by Megatron Core `core_v0.19.2` with its local layer spec (no Transformer Engine, no apex), one step traced with Python stacks and one without. The walkthrough is [day13/GUIDE.md](../../day13/GUIDE.md), and the notes template is [day13/notes.md](../../day13/notes.md).

## Files

| File | What it is | Needs |
| --- | --- | --- |
| `rank_mapper.py` | rank → (TP, DP, PP) and every group kind; reference + **YOU WRITE** `my_coords` / `my_groups` (TODO 1–2); compares with Megatron's `RankGenerator` with `--megatron PATH` | Python (torch only with `--megatron`) |
| `my_step.py` | **YOU WRITE** `loss_func` / `forward_step_func` (TODO 3–4), used by `train_gpt.py --impl mine` | — |
| `train_gpt.py` | the Megatron job: process groups (checked against the mapper inside the job), `GPTModel`, same full weights in every layout, `get_forward_backward_func`, Adam, two profiled steps | torch, Megatron on `PYTHONPATH`, `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| `analyze_trace.py` | ties every NCCL kernel to its group, phase, thread, stream and Megatron frame; reads the F/B order and the compute stream's idle time | Python stdlib (runs on Windows) |
| `compare_runs.py` | losses and step-0 gradient norms vs the 1-GPU job, P2P shapes, NCCL counts vs the derived counts, 1F1B order | Python stdlib |
| `topo_check.py` | parses `nvidia-smi topo -m`, picks the fastest pair and a 4-GPU order with TP pairs on the best links | Python |
| `find_symbols.py` | file:line of the Block 3 stops in your Megatron checkout | Python |
| `setup_megatron.sh` | clones `core_v0.19.2` to `/workspace/Megatron-LM`, checks the commit, installs `einops` | network |
| `run_server_once.sh` | every stage once, checked, archived with a `SHA256SUMS` manifest; never stops the instance | four GPUs |
| `download_results.ps1` | Windows: download, check the archive's SHA-256, extract, re-hash every manifest entry | OpenSSH |
| `stall_probe.py` | GUIDE Appendix C: the `tp2pp2` job for hundreds of steps with host timestamps on every F/B/P2P call, GC callbacks, optional profiler pattern, manual GC, NUMA pinning, allocator counters | four GPUs |
| `run_stall_matrix.sh` | Appendix C.2: six probe cases (profiler release, manual GC, NUMA pinning), about 6 min | four GPUs |
| `host_sampler.py`, `stall_correlate.py`, `run_sampled.sh` | Appendix C.3: `/proc` thread states, cgroup/PSI and `nvidia-smi` sampled beside the probe, joined with its stalls (`CATCH=1` adds the catcher) | four GPUs (correlate: stdlib) |
| `stall_catcher.py` | Appendix C.3: gdb stacks of every thread in the middle of a stall (probe `--allow-ptrace`) | gdb |
| `nsys_trace.py` | Appendix C.6: reads an `nsys export --type sqlite` capture into `analyze_trace.py`'s event shape, plus every thread's long CUDA API / OS-runtime calls | Python stdlib |
| `run_nsys_check.sh` | Appendix C.6: PyTorch profiler vs `--profiler nsys`, N runs each, then `stall_probe.py` under nsys | four GPUs, `nsys` |
| `run_stall_fix_check.sh` | Appendix C.2: pre-fix vs fixed `train_gpt.py`, N runs each (needs `train_gpt_before.py`, a copy of the pre-fix file) | four GPUs |

## Commands

On the instance, from `/workspace/AI-Infra-Learning`:

```bash
bash experiments/day13_megatron/setup_megatron.sh
```

```bash
RID=manual_$(date -u +%Y%m%dT%H%M%SZ); bash experiments/day13_megatron/run_server_once.sh "$RID"
```

About 4.5 minutes. Options: `--impl mine` (your `my_step.py`), `--mapper mine` (your `rank_mapper.py`), `--quick` (only the 1-GPU and the 4-GPU job), `--no-sp` (skip the sequence-parallel comparison), `--megatron PATH`.

Single jobs (the runner does the same with the logged GPU choice):

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/day13_megatron/train_gpt.py --tp 2 --pp 2 --run-id try1
python experiments/day13_megatron/analyze_trace.py outputs/day13_megatron/run_try1/tp2pp2 --megatron /workspace/Megatron-LM
```

With Nsight Systems instead of the PyTorch profiler (only the timing step is recorded; `analyze_trace.py` reads the `nsys_rank*.sqlite` files):

```bash
bash experiments/day13_megatron/run_nsys_check.sh n1 3 0     # 3 PyTorch-profiler + 3 nsys runs; last arg: probe steps under nsys (0 = skip)
```

On Windows, after the run:

```powershell
experiments/day13_megatron/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>
```

## Outputs

`outputs/day13_megatron/session_<RID>/` holds one log per stage (`environment`, `topology`, `symbols`, `mapper`, `run_<layout>`, `analyze_<layout>`, `compare`, `verify_core`), `commands.txt`, `gpu_choice.txt`, `compare.json`, `status.txt` and `SHA256SUMS`. `outputs/day13_megatron/run_<RID>/<layout>/` holds `rank<r>.json`, `summary.json`, `trace_rank<r>.json.gz` (stack step), `timing_rank<r>.json.gz` (timing step) and `trace_analysis.json`. Open a trace in <https://ui.perfetto.dev> (drag the `.json.gz` in). The archive is `outputs/day13_exports/day13_<RID>.tar.gz` (about 27 MB) with a `.sha256` file next to it.

## Deviations from a stock Megatron run, on purpose

- No `megatron/training` loop and no `pretrain_gpt.py`: `train_gpt.py` is that loop reduced to its Megatron Core calls (GUIDE §2 maps each piece to `pretrain_gpt.py` / `training.py`).
- No Megatron DDP wrapper (DP = 1), plain `torch.optim.Adam`. With SP, the TP all-reduce of the LayerNorm/bias gradients that `finalize_model_grads` would do is done by hand (`allreduce_sequence_parallel_grads`).
- With `--sp` only: `SequenceParallelTorchLayerNorm` replaces the local spec's torch LayerNorm fallback, which asserts against SP without apex/TE. It only flags the parameters, as apex's `FusedLayerNorm` does.
- `schedules.backward_step` is wrapped to add a profiler range; numerics are unchanged.
- After each trace export the profiler is deleted and `gc.collect()` runs, outside the timed step. `torch.profiler.profile` is a reference cycle, and without this the GC freed it inside F1 of the next step, a 50–95 ms stall (GUIDE Appendix C.2).
- All weights come from `canonical_init` (full tensor per global name, then this rank's TP shard), so every layout starts from the same model.
