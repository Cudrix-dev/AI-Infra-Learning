# Day 12: two-stage pipeline parallelism, GPipe and non-interleaved 1F1B

A 4-block residual MLP split over two GPUs by depth: stage 0 (rank 0) runs blocks 0–1, stage 1 (rank 1) runs blocks 2–3. Each microbatch sends one activation forward and one activation gradient back. Two schedules run the same microbatches: GPipe (all forwards, then all backwards) and Megatron's non-interleaved 1F1B (warmup, steady one-forward-one-backward, cooldown). A CPU simulator predicts each stage's timeline, idle time, bubble and peak in-flight microbatches. The GPU checker compares outputs and parameter gradients with the same model on one GPU, before any optimizer step, and checks that each rank's real op order and in-flight count match the simulator.

Walkthrough: [`day12/GUIDE.md`](../../day12/GUIDE.md). Write-up template: [`day12/notes.md`](../../day12/notes.md).

## Experiment host

Same template as Days 02–11: 2 × A100-SXM4 with NVLink, torch 2.14.1+cu130, NCCL 2.30.7. One process per GPU via `torchrun`. The single-GPU baseline uses GPU 0 alone. No new package. `pp_sim.py` and `find_symbols.py` need no torch and run on Windows. This folder imports no code from earlier days.

On a fresh rental, upload `README.md`, `requirements.txt`, `day12/` and this folder, then `ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv`.

## Files

| File | Purpose |
| --- | --- |
| `pp_sim.py` | CPU schedule simulator. Assumptions in its docstring: fixed per-stage order, t_F = 1, t_B = 2, comm 0, buffered or rendezvous sends. Grids, idle, bubble, in-flight peak/mean/residency, Megatron-style P2P programs, `--sweep`, `--stage-times`, a deadlock model, `--check` against the GUIDE's hand-worked numbers |
| `boundary_demo.py` | Autograd across a cut with scalars (one CPU process): what is saved, what crosses, three mistakes |
| `pp_common.py` | `Shapes` (seq 256, effective batch 8, d 512, ffn 2048, 4 blocks, 2 stages), `make_problem`, `Block`, `stage_blocks`, `microbatches`, `mb_loss` (sum / N), `compare`; provided `P2P` (Megatron's six helpers on `batch_isend_irecv`), `Stage`, `EventLog` |
| `my_pp.py` | **Yours (YOU WRITE).** `forward_step`, `backward_step`, `run_gpipe`, `run_1f1b`. While a TODO remains, the checker exits with code 3 before any communication |
| `pp_ref.py` | Reference for the same four functions |
| `baseline_1gpu.py` | The whole model on one GPU: gradient accumulation for M = 1, 2, 4, 8; saves y, loss and all gradients to `outputs/day12_pp/tensors/` |
| `check_pp.py` | `--test boundary` (Step 1, one microbatch) or `--schedules gpipe,1f1b --microbatches 1,2,4,8`: parity against accumulation with the same M and against the full batch, order and peak vs the simulator, P2P label matching, CUDA peak per rank, GPipe vs 1F1B bitwise; `--controls`; `--device cpu` (gloo) |
| `pp_controls.py` | `detached_input`, `lifo_backward`, `mean_per_microbatch` and the exact checks each must fail |
| `find_symbols.py` | file:line of the Block 3 reading list (`schedules.py`, `p2p_communication.py`) in a Megatron-LM checkout |
| `run_server_once.sh` | Every stage once, logged, output-checked, archived with SHA-256 in `outputs/day12_exports/` (`--impl ref|mine`, `--quick`) |
| `download_results.ps1` | From Windows: download, check the archive SHA-256, extract, re-check every file against the manifest |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
D12=experiments/day12_pipeline_parallel
python $D12/pp_sim.py --check                                                       # SIMULATOR CHECK: PASSED
python $D12/pp_sim.py                                                               # the P=2, M=4 grids
python $D12/boundary_demo.py                                                        # BOUNDARY DEMO: PASSED
CUDA_VISIBLE_DEVICES=0 python $D12/baseline_1gpu.py --run-id try1                   # BASELINE: SAVED
torchrun --standalone --nproc_per_node=2 $D12/check_pp.py --impl ref --schedules gpipe,1f1b \
    --microbatches 1,2,4,8 --controls --baseline outputs/day12_pp/tensors/baseline_float32_try1.pt --run-id try1
```

Replace `--impl ref` with `--impl mine` for your code. Use `--test boundary` after Step 1. Use `--device cpu` to check the logic without GPUs (no memory numbers).

Everything at once (no automatic shutdown; about 30 s):

```bash
bash $D12/run_server_once.sh "$RID" --impl ref
```

```bash
bash $D12/run_server_once.sh "${RID}_mine" --impl mine
```

Then, from Windows PowerShell in the project root:

```powershell
experiments/day12_pipeline_parallel/download_results.ps1 -Remote root@HOST -Port PORT -RunId RUN_ID
```

Outputs go to `outputs/day12_pp/` (git-ignored): `baseline_float32_<RID>.json`, `boundary_<impl>_<RID>.json`, `parity_<impl>_<RID>.json` (every check per case, per-rank orders, P2P calls, labels, CUDA peaks, control verdicts), `sim_*_<RID>.json`, `events_<RID>/<impl>_<schedule>_M<M>.jsonl` (both ranks' events in order), and `session_<RID>/` with every log, `environment.log`, a project snapshot and `SHA256SUMS`. The baseline tensors (about 185 MB) are not archived. They are reproducible from the seed, and the baseline JSON records the problem's SHA-256.

## Notes

- **Pass criterion.** For `y`, `loss` and each of the 16 parameter gradients: `max|pp − ref| / max|ref| ≤ 1e-5` (FP32) against **both** single-GPU accumulation with the same M and the full batch. In addition, each rank's F/B order and peak in-flight count must equal the simulator's, and the P2P labels must pair up. Bitwise flags are reported. The pre-test was bitwise equal to same-M accumulation for `y` and all gradients.
- **TF32 is off** (`torch.set_float32_matmul_precision("highest")`).
- **Labels are not transmitted.** The `mb` arguments of the `P2P` helpers are logged only, and the checker compares sender and receiver sequences. Only tensors cross the wire.
- **CUDA peak** = `max_memory_allocated() − memory_allocated()` around the schedule on each rank, with parameters, `.grad` buffers, inputs, targets and the output buffer allocated before. NCCL's own buffers are not included. The first case in a process can read about 16 MiB higher (GUIDE Appendix A).
- **GPU timestamps** in the event logs are CUDA events on each rank's compute stream since a barrier. They show order and gaps. They are not a benchmark.
