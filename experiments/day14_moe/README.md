# Day 14: a top-1 MoE layer with expert parallelism (EP=4) on four GPUs

The Day 01 GPT (small preset: d 128, 2 layers, 8 × 64 tokens; medium for timing) with every MLP replaced by a top-1 MoE layer of E = 4 experts. One expert per GPU at EP=4, dropless, tokens moved with uneven `all_to_all_single`. Everything is checked against a one-GPU baseline on the same host. FP32 with TF32 off. The walkthrough is [day14/GUIDE.md](../../day14/GUIDE.md), and the notes template is [day14/notes.md](../../day14/notes.md).

## Files

| File | What it is | Needs |
| --- | --- | --- |
| `moe_common.py` | config/presets, deterministic per-name init, Day 01 batches, forced routing patterns, send-matrix and load metrics, `compare()`, process groups (`layout_ranks`, `build_groups`), `spawn_local` for CPU runs | torch |
| `moe_ref.py` | the reference for the five TODOs (`top1_route`, `permute`, `unpermute`, `exchange_counts`, `AllToAllV`, `finish_grads`) plus shared helpers (sort_chunks, aux/z loss, AutoScaler, capacity mask, split checks) | torch |
| `my_moe.py` | **YOU WRITE**: the same six names, stubs that raise `NotImplementedError` (TODO 1–5, about 1 h 35 min) | — |
| `moe_model.py` | `MoELayer` (route → counts → dispatch → experts → combine → unpermute), the transformer around it, `build_model` | torch |
| `moe_controls.py` | five negative controls (wrong inverse ×2, expert grads summed over EP, no 1/DP scale, recv = send) and the failures each must produce | torch |
| `check_local.py` | the GUIDE's hand-worked examples (gating, permutation, capacity, aux loss) plus split checks and a CPU-simulated EP=4 / EP=8 dispatch | torch, CPU |
| `baseline_1gpu.py` | the oracle: EP=1, all experts on one device, every routing pattern; logits, losses, dx and every gradient saved to a `.pt` | 1 GPU (or CPU) |
| `check_ep.py` | EP parity vs the baseline (`--tests parity`), the a2a forward/backward on labelled rows (`--tests dispatch`), the controls (`--impl controls`), EP=2 × EDP=2 (`--ep 2`), Megatron's local aux scope (`--aux-scope local`) | 4 GPUs (or `--cpu --nproc 4`) |
| `route_stats.py` | Exp 1: tokens per expert, capacity drops (global vs per-source C), drop-and-pad rows, receive rows per EP rank, EP=8 [simulated]; optional `--aux-train N` | 1 GPU (or CPU) |
| `bench_skew.py` | Exp 3: one MoE layer, balanced / hot / all_to_one, per-rank phase times, receive rows, slowest rank; `--ep 1` reference; optional `--trace` | 4 GPUs / 1 GPU |
| `find_symbols.py` | file:line of the Block 3 stops in your Megatron checkout; exit 1 if any is missing | Python |
| `run_server_once.sh` | every stage once, checked, archived with a `SHA256SUMS` manifest; never stops the instance | four GPUs |
| `download_results.ps1` | Windows: download, check the archive's SHA-256, extract, re-hash every manifest entry | OpenSSH |

There is no separate group-layout script. `moe_common.layout_ranks` *is* the layout (rank = ep_rank + EP · edp_rank, Megatron's expert order with TP = PP = 1), and `check_ep.py --ep 2` exercises it.

## Commands

Upload with `scp` as on Days 11–13 (not git), then on the instance, from `/workspace/AI-Infra-Learning`:

```bash
RID=manual_$(date -u +%Y%m%dT%H%M%SZ); bash experiments/day14_moe/run_server_once.sh "$RID"
```

The reference pre-test (`pretest2`, 4 × A100-SXM4-40GB) took about 2 minutes in full mode, mostly process start-ups. The flags are:

- `--core` runs only the required stages.
- `--quick` runs check_local, the baseline, dispatch and parity.
- `--impl mine` uses your `my_moe.py`. Parity is still checked against the reference baseline.
- `--megatron PATH` points the symbols stage at a Megatron checkout.

Megatron is only read, never run. If it isn't there, the symbols stage is skipped.

Single stages:

```bash
python experiments/day14_moe/check_local.py                      # CPU, seconds
CUDA_VISIBLE_DEVICES=0 python experiments/day14_moe/baseline_1gpu.py --run-id try1
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/day14_moe/check_ep.py \
    --baseline outputs/day14_moe/tensors/baseline_small_try1.pt
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/day14_moe/bench_skew.py
```

On your PC (no GPU), the same checks run on gloo with four local processes. The timings are meaningless:

```powershell
python experiments/day14_moe/check_local.py --impl mine
python experiments/day14_moe/baseline_1gpu.py --cpu --run-id cpu1
python experiments/day14_moe/check_ep.py --cpu --nproc 4 --impl mine --baseline outputs/day14_moe/tensors/baseline_small_cpu1.pt
```

After the run, on Windows:

```powershell
experiments/day14_moe/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>
```

## Exit codes and final lines

Each check ends with one line that `verify_core` reads:

- `CHECK_LOCAL (impl): N passed, M failed, K incomplete: PASSED`
- `BASELINE written: … (sha256 …)`
- `DISPATCH (…): PASSED`
- `PARITY (…): PASSED`
- `CONTROLS (…): PASSED`, meaning every control failed exactly as predicted and the presorted blind spot failed nothing
- `BENCH_SKEW: done`

The exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | passed |
| 1 | a check failed |
| 2 | bad usage |
| 3 | `my_moe.py` still has TODOs. The script prints `INCOMPLETE (TODOs left)`. |

## Outputs

`outputs/day14_moe/session_<RID>/` holds:

- one log per stage
- the JSONs: `check_local`, `route_stats`, `ep4_dispatch`, `ep4_parity`, `ep4_controls`, `bench_ep1`, `bench_ep4`, plus any optional ones
- `commands.txt`, `optional_status.txt` and `status.txt`
- `CORE_SUCCESS.txt` / `SUCCESS.txt`
- `SHA256SUMS`
- a project snapshot

`outputs/day14_moe/run_<RID>/` holds:

- the baseline `.pt` (about 38 MB: every gradient for six routing patterns)
- `traces/trace_ep4_large_<pattern>_rank<r>.json` (optional `bench_trace` stage). Open these in <https://ui.perfetto.dev>.

The archive is `outputs/day14_exports/day14_<RID>.tar.gz`, with a `.sha256` next to it. Nothing writes full medium-size tensors, which keeps the run well inside the 60 GB disk.

## Deviations from Megatron, on purpose

- **Router weight.** The combine weight is the selected entry of the full softmax, the same as `moe_router_pre_softmax=True`. Megatron's default (`False`) takes the softmax over the top-k scores. With k = 1 that is softmax of one number = 1.0, so the router would get no task-loss gradient (GUIDE P1, Block 3 Stop 1).
- **Aux-loss counts.** The aux loss counts tokens over the global batch: the counts are all-reduced over dense DP. That matches the one-GPU baseline. Megatron's default `aux_loss` reduces them over `tp_cp_group` only, so they are local at TP = CP = 1. `--aux-scope local` reproduces that and is expected to differ (P9).
- **Probs.** Probs stay at the source rank and multiply after the combine. Megatron sends them through a second all_to_all and applies them inside the expert, before fc2.
- **Count exchange.** Counts are exchanged with an `all_to_all` of an [EP, E_local] matrix. Megatron all-gathers every rank's full count vector over `tp_ep_group`. In both cases there is one device-to-host copy (`.tolist()`) before the token all_to_all.
- **Experts.** Experts are a Python loop (`SequentialMLP`-style), not grouped GEMM.
- **Gradient averaging.** No Megatron DDP: dense grads are all-reduced over the world and divided by DP. Expert grads are divided by DP and then all-reduced over EDP, the same as Megatron's `expert_gradient_scaling_factor = 1/dp`.
