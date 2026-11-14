# Day 11: tensor-parallel MLP (column → GELU → row) with explicit autograd communication

A two-GPU MLP: a column-parallel first linear layer, an elementwise GELU on the sharded activation, and a row-parallel second linear layer. The two communication points are custom `torch.autograd.Function`s: `CopyToTP` (f, identity forward, all-reduce backward) and `ReduceFromTP` (g, all-reduce forward, identity backward). Acceptance is parity with the same MLP on one GPU: forward output, input gradient, and every weight and bias gradient, with identical tensors and the same loss. Four negative controls show what each common mistake does. Megatron Core's own layers run through the same check as an option.

Walkthrough: [`day11/GUIDE.md`](../../day11/GUIDE.md). Write-up template: [`day11/notes.md`](../../day11/notes.md).

## Experiment host

Same template as Days 02–10: 2 × A100-SXM4 with NVLink, torch 2.14.1+cu130, NCCL 2.30.7. One process per GPU via `torchrun`. The single-GPU baseline uses GPU 0 alone. No new package. The optional Megatron stage needs a Megatron-LM source checkout (`core_v0.19.2`) on `PYTHONPATH`, with no pip install, Transformer Engine or apex. This folder imports no code from earlier days.

On a fresh rental, upload `README.md`, `requirements.txt`, `day11/` and this folder, then `ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv`.

## Files

| File | Purpose |
| --- | --- |
| `tiny_example.py` | Two "ranks" in one CPU process with integer matrices: what each rank owns, forward and backward, exact checks |
| `tp_common.py` | `Shapes` (seq 256, batch 2, d 512, ffn 2048), `make_problem` (seed → float64 on CPU → cast), `dense_forward`, `loss_fn` (MSE), `compare`, `CollectiveCounter` |
| `my_tp.py` | **Yours (YOU WRITE).** `shard_column_parallel`, `shard_row_parallel`, `CopyToTP`, `ReduceFromTP`, `ColumnParallelLinear.forward`, `RowParallelLinear.forward`. Until filled, the scripts exit with code 3 |
| `tp_ref.py` | **Reference** for the same names, plus the shared `TPMLPBlock`, `TPMLP`, `build_tp_model` |
| `baseline_1gpu.py` | Dense `nn.Linear` MLP on one GPU; saves inputs, parameters, y, dx and gradients to `outputs/day11_tp/tensors/` |
| `check_tp_parity.py` | `--test unit` (shards, f, g on values `r+1`) and `--test parity` (gather shards, compare with the baseline, count collectives per phase, `--controls`). `--impl ref|mine|megatron`, `--device cuda|cpu` |
| `tp_controls.py` | `no_dx_allreduce`, `g_bwd_allreduce`, `avg_dx`, `bias_before_reduce`, and the checks each must fail for 1 and 2 blocks |
| `megatron_impl.py` | Optional: the same blocks from Megatron Core's `ColumnParallelLinear` / `RowParallelLinear` |
| `find_symbols.py` | `file:line` of the Block 3 reading list in a Megatron-LM checkout (no torch needed) |
| `run_server_once.sh` | Every stage once, logged, output-checked, archived with SHA-256 in `outputs/day11_exports/` |
| `download_results.ps1` | From Windows: download, check the archive SHA-256, extract, re-check every file against the manifest |

## Usage (A100 project root, `.venv` active)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
D11=experiments/day11_tensor_parallel
python $D11/tiny_example.py                                                                   # TINY EXAMPLE: PASSED
torchrun --standalone --nproc_per_node=2 $D11/check_tp_parity.py --impl ref --test unit       # UNIT TESTS: PASSED
CUDA_VISIBLE_DEVICES=0 python $D11/baseline_1gpu.py --run-id try1                             # BASELINE: SAVED
torchrun --standalone --nproc_per_node=2 $D11/check_tp_parity.py --impl ref --controls \
    --baseline outputs/day11_tp/tensors/baseline_float32_b1_try1.pt --run-id try1             # PARITY ... PASSED, CONTROLS: ALL DETECTED AS PREDICTED
```

Variants: `baseline_1gpu.py --blocks 2` (two stacked MLPs) and `--dtype float64`. Pass the matching baseline file to the checker. Replace `--impl ref` with `--impl mine` for your code. Use `--device cpu` (gloo) to test the logic without GPUs.

Everything at once (no automatic shutdown):

```bash
bash $D11/run_server_once.sh "$RID" --impl ref [--megatron /workspace/Megatron-LM]    # ~50 s (+25 s Megatron)
bash $D11/run_server_once.sh "${RID}_mine" --impl mine                                 # ~40 s
```

Then, from Windows PowerShell in the project root:

```powershell
experiments/day11_tensor_parallel/download_results.ps1 -Remote root@HOST -Port PORT -RunId RUN_ID
```

Outputs go to `outputs/day11_tp/` (git-ignored): `baseline_<dtype>_b<N>_<RID>.json`, `unit_<impl>_<RID>.json`, `parity_<impl>_<dtype>_b<N>_<RID>.json` (every check, every collective, per-rank shapes, the control verdicts), and `session_<RID>/` with every log, `environment.log`, a project snapshot and `SHA256SUMS`. The baseline tensors in `outputs/day11_tp/tensors/` are not archived. They are reproducible from the seed, and the baseline JSON records the SHA-256 of the input and parameter tensors.

## Notes

- **Pass criterion.** For every tensor, `max|tp − 1gpu| / max|1gpu| ≤ 1e-5` (FP32) or `1e-12` (FP64). The MSE gradients are small (the mean divides by 262,144), so an absolute tolerance would mean nothing. In addition, `y`, `dx` and each `b2` gradient must be bitwise identical across ranks, and each block issues exactly one all-reduce in forward and one in backward. Bitwise equality with the baseline is reported but not required.
- **`ratio` = ⟨tp, ref⟩ / ⟨ref, ref⟩.** 1.0000 when correct. A clean power of the TP size means a scale bug, anything else a missing or misplaced term.
- **TF32 is off** (`torch.set_float32_matmul_precision("highest")`), so FP32 differences are summation order only.
- **The collective counter** wraps `torch.distributed.all_reduce` (and the gather/scatter collectives) at the module attribute. Code must call `dist.all_reduce(...)`, not a name imported before the wrap. The checker's own gathers are not counted.
