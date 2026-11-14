#!/usr/bin/env bash
# Day 11: run every Day 11 stage once on TWO GPUs, check the outputs, and pack a checksummed archive.
# Run from Linux; no driver installation and no automatic shutdown (stop the instance yourself after downloading).
#   bash experiments/day11_tensor_parallel/run_server_once.sh [run-id] [--impl ref|mine] [--megatron PATH]
#   --impl ref (default): tiny example, unit tests, then for fp32 x 1 block, fp32 x 2 blocks and fp64 x 1 block:
#                         the single-GPU baseline and the 2-GPU parity check; the four negative controls on the
#                         two fp32 cases.
#   --impl mine: the same stages for my_tp.py (no controls: they are built from the reference).
#   --megatron PATH: optional extra stage, Megatron Core's layers on the fp32 baselines (PATH = Megatron-LM checkout).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
IMPL=ref
MEGATRON=
if [[ $# -gt 0 && "$1" != --* ]]; then RID=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --impl) IMPL=${2:-}; shift 2 ;;
        --megatron) MEGATRON=${2:-}; shift 2 ;;
        *) echo 'Usage: run_server_once.sh [run-id] [--impl ref|mine] [--megatron PATH]' >&2; exit 2 ;;
    esac
done
[[ "$IMPL" == ref || "$IMPL" == mine ]] || { echo '--impl must be ref or mine' >&2; exit 2; }
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
[[ -z "$MEGATRON" || -d "$MEGATRON/megatron/core" ]] || { echo "--megatron: $MEGATRON/megatron/core not found" >&2; exit 2; }
OUT="$PROJECT/outputs/day11_tp"
SESSION="$OUT/session_$RID"
mkdir -p "$OUT" "$PROJECT/outputs/day11_exports"
mkdir "$SESSION" # Refuse to overwrite an existing session.
ARCHIVE="$PROJECT/outputs/day11_exports/day11_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY11_SESSION="$SESSION" DAY11_OUT="$OUT" DAY11_RID="$RID" DAY11_IMPL="$IMPL" DAY11_MEGATRON="$MEGATRON"
printf 'run_id=%s\nimpl=%s\nmegatron=%s\nstarted_utc=%s\n' "$RID" "$IMPL" "${MEGATRON:-none}" "$(date -u +%FT%TZ)" \
    > "$SESSION/commands.txt"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY11_OUT']), os.environ['DAY11_RID']
session = Path(os.environ['DAY11_SESSION'])
# JSON results and the session folder; the baseline tensors (outputs/day11_tp/tensors/) stay on the server.
roots = sorted(p for p in out.iterdir() if p.name.endswith('_' + rid) or p.stem.endswith('_' + rid))
manifest = session / 'SHA256SUMS'
with manifest.open('w') as f:
    for root in roots:
        for p in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if p.is_file() and p != manifest and p.name != 'archive_members.txt':
                with p.open('rb') as data:
                    digest = hashlib.file_digest(data, 'sha256').hexdigest()
                f.write(f'{digest}  {p.relative_to(out).as_posix()}\n')
(session / 'archive_members.txt').write_text(''.join(p.name + '\n' for p in roots))
PY
    if [[ $? -ne 0 ]]; then echo "Manifest failed; download $OUT manually."; exit 1; fi
    tar -czf "$ARCHIVE" -C "$OUT" -T "$SESSION/archive_members.txt" \
        && (cd "$(dirname "$ARCHIVE")" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
    if [[ $? -ne 0 ]]; then echo "Packaging failed; download $OUT manually."; exit 1; fi
    echo "Archive: $ARCHIVE"
    echo "Run exit code: $rc (0 = all requested stages and output checks passed)"
    exit "$rc"
}
trap finish EXIT

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1} OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" >> "$SESSION/commands.txt"

run_logged() {
    local name=$1
    shift
    printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$name"
    printf '%s: %s\n' "$name" "$*" >> "$SESSION/commands.txt"
    "$@" 2>&1 | tee "$SESSION/$name.log"
}

df -h /workspace | tee "$SESSION/disk_before.txt"
run_logged environment python - <<'PY'
import os, platform, shutil, subprocess, sys, torch
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expected exactly two visible GPUs'
s = Path(os.environ['DAY11_SESSION'])
print('Platform:', platform.platform())
print('Python:', platform.python_version(), sys.executable)
print('Torch:', torch.__version__, '| CUDA build:', torch.version.cuda, '| NCCL:', '.'.join(map(str, torch.cuda.nccl.version())))
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
print('CONTAINER_ID:', os.environ.get('CONTAINER_ID', '?'), '| hostname:', platform.node())
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(f'cuda:{i}', p.name, f'{p.total_memory / 2**30:.1f} GiB', f'sm_{p.major}{p.minor}', f'{p.multi_processor_count} SMs')
print('P2P cuda:0 <-> cuda:1:', torch.cuda.can_device_access_peer(0, 1), torch.cuda.can_device_access_peer(1, 0))
def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr
print(sh('nvidia-smi', '--query-gpu=index,name,driver_version,pstate,memory.used,memory.total,utilization.gpu', '--format=csv'))
print(sh('nvidia-smi', 'topo', '-m'))
print(sh('nvidia-smi', 'nvlink', '-s'))
snapshot = s / 'project_snapshot'
for folder in ['day11', 'experiments/day11_tensor_parallel']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
PY

D11=experiments/day11_tensor_parallel
TORCHRUN=(python -m torch.distributed.run --standalone --nproc_per_node=2)
run_logged tiny_example python $D11/tiny_example.py
run_logged unit "${TORCHRUN[@]}" $D11/check_tp_parity.py --impl "$IMPL" --test unit --run-id "$RID"

CTRL=()
[[ "$IMPL" == ref ]] && CTRL=(--controls)
for CASE in float32:1 float32:2 float64:1; do
    DT=${CASE%%:*}; NB=${CASE##*:}
    # The baseline sees ONE GPU (physical GPU 0 of the pair); the parity run uses both.
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} run_logged "baseline_${DT}_b${NB}" \
        python $D11/baseline_1gpu.py --run-id "$RID" --dtype "$DT" --blocks "$NB"
    ARGS=(--impl "$IMPL" --test parity --baseline "$OUT/tensors/baseline_${DT}_b${NB}_${RID}.pt" --run-id "$RID")
    [[ "$DT" == float32 ]] && ARGS+=("${CTRL[@]}")
    run_logged "parity_${DT}_b${NB}" "${TORCHRUN[@]}" $D11/check_tp_parity.py "${ARGS[@]}"
done

run_logged verify_core python - <<'PY'
import json, os
from pathlib import Path
out, rid, s, impl = Path(os.environ['DAY11_OUT']), os.environ['DAY11_RID'], Path(os.environ['DAY11_SESSION']), os.environ['DAY11_IMPL']
assert 'TINY EXAMPLE: PASSED' in (s / 'tiny_example.log').read_text()
assert 'UNIT TESTS: PASSED' in (s / 'unit.log').read_text()
for dt, nb in [('float32', 1), ('float32', 2), ('float64', 1)]:
    log = (s / f'parity_{dt}_b{nb}.log').read_text()
    assert f'PARITY ({impl}, {dt}, {nb} block(s)): PASSED' in log, (dt, nb)
    r = json.loads((out / f'parity_{impl}_{dt}_b{nb}_{rid}.json').read_text())
    assert r['passed'] and r['exit_code'] == 0
    b = json.loads((out / f'baseline_{dt}_b{nb}_{rid}.json').read_text())
    assert b['repeat_bitwise'], 'baseline not reproducible on its own GPU'
    if impl == 'ref' and dt == 'float32':
        assert 'CONTROLS: ALL DETECTED AS PREDICTED' in log, (dt, nb)
        assert len(r['controls']) == 4 and all(v['as_predicted'] for v in r['controls'].values())
print('OUTPUT CHECKS PASSED: tiny example, unit tests, parity fp32 x1, fp32 x2, fp64 x1'
      + (', 8 control runs detected as predicted' if impl == 'ref' else ''))
PY
printf 'Required stages and output checks passed.\n' > "$SESSION/CORE_SUCCESS.txt"

if [[ -n "$MEGATRON" ]]; then
    ( cd "$MEGATRON" && printf 'megatron_checkout=%s\nmegatron_describe=%s\nmegatron_commit=%s\n' "$MEGATRON" \
        "$(git describe --tags 2>/dev/null || echo '?')" "$(git rev-parse HEAD 2>/dev/null || echo '?')" ) \
        | tee -a "$SESSION/commands.txt" > "$SESSION/megatron_version.txt"
    for NB in 1 2; do
        PYTHONPATH="$MEGATRON${PYTHONPATH:+:$PYTHONPATH}" CUDA_DEVICE_MAX_CONNECTIONS=1 run_logged "megatron_float32_b${NB}" \
            "${TORCHRUN[@]}" $D11/check_tp_parity.py --impl megatron --test parity \
            --baseline "$OUT/tensors/baseline_float32_b${NB}_${RID}.pt" --run-id "$RID"
        grep -q "PARITY (megatron, float32, ${NB} block(s)): PASSED" "$SESSION/megatron_float32_b${NB}.log"
    done
    echo 'Megatron stage passed.' >> "$SESSION/CORE_SUCCESS.txt"
fi
nvidia-smi --query-gpu=index,pstate,temperature.gpu,memory.used --format=csv | tee "$SESSION/gpu_after.txt"
df -h /workspace | tee "$SESSION/disk_after.txt"
printf 'All requested stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
