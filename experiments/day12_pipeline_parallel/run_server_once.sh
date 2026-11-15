#!/usr/bin/env bash
# Day 12: run every Day 12 stage once on TWO GPUs, check the outputs, and pack a checksummed archive.
# Run from Linux; no driver installation and no automatic shutdown (stop the instance yourself after downloading).
#   bash experiments/day12_pipeline_parallel/run_server_once.sh [run-id] [--impl ref|mine] [--quick]
#   Stages: environment, simulator self-check + example + sweep + unequal-stage + deadlock demo (CPU),
#           boundary demo (CPU), single-GPU baseline for M = 1,2,4,8, the Step 1 boundary check,
#           the two-GPU parity matrix (GPipe and 1F1B x M = 1,2,4,8), and for --impl ref the three negative controls.
#   --impl mine: the same stages for my_pp.py (no controls: they are built from the reference).
#   --quick: parity at M = 4 only (both schedules).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
IMPL=ref
MS=1,2,4,8
if [[ $# -gt 0 && "$1" != --* ]]; then RID=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --impl) IMPL=${2:-}; shift 2 ;;
        --quick) MS=4; shift ;;
        *) echo 'Usage: run_server_once.sh [run-id] [--impl ref|mine] [--quick]' >&2; exit 2 ;;
    esac
done
[[ "$IMPL" == ref || "$IMPL" == mine ]] || { echo '--impl must be ref or mine' >&2; exit 2; }
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
OUT="$PROJECT/outputs/day12_pp"
SESSION="$OUT/session_$RID"
mkdir -p "$OUT" "$PROJECT/outputs/day12_exports"
mkdir "$SESSION" # Refuse to overwrite an existing session.
ARCHIVE="$PROJECT/outputs/day12_exports/day12_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY12_SESSION="$SESSION" DAY12_OUT="$OUT" DAY12_RID="$RID" DAY12_IMPL="$IMPL" DAY12_MS="$MS"
printf 'run_id=%s\nimpl=%s\nmicrobatches=%s\nstarted_utc=%s\n' "$RID" "$IMPL" "$MS" "$(date -u +%FT%TZ)" \
    > "$SESSION/commands.txt"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY12_OUT']), os.environ['DAY12_RID']
session = Path(os.environ['DAY12_SESSION'])
# JSON results, event logs and the session folder; the baseline tensors (outputs/day12_pp/tensors/) stay on the server.
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
s = Path(os.environ['DAY12_SESSION'])
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
for folder in ['day12', 'experiments/day12_pipeline_parallel']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
PY

D12=experiments/day12_pipeline_parallel
TORCHRUN=(python -m torch.distributed.run --standalone --nproc_per_node=2)
# CPU stages first: the simulator and the single-process boundary demo.
run_logged sim_check python $D12/pp_sim.py --check
run_logged sim_example python $D12/pp_sim.py --show-programs --json "$OUT/sim_example_$RID.json"
run_logged sim_sweep python $D12/pp_sim.py --sweep 1,2,4,8,16 --json "$OUT/sim_sweep_$RID.json"
run_logged sim_unequal python $D12/pp_sim.py --stage-times 1:2,2:4 --json "$OUT/sim_unequal_$RID.json"
run_logged sim_deadlock python $D12/pp_sim.py --schedule 1f1b --comm-style naive --send rendezvous --show-programs
run_logged boundary_demo python $D12/boundary_demo.py

# The baseline sees ONE GPU (physical GPU 0 of the pair); the pipeline uses both.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} run_logged baseline \
    python $D12/baseline_1gpu.py --run-id "$RID" --microbatches 1,2,4,8
BASE="$OUT/tensors/baseline_float32_${RID}.pt"
run_logged boundary "${TORCHRUN[@]}" $D12/check_pp.py --impl "$IMPL" --test boundary --baseline "$BASE" --run-id "$RID"
CTRL=()
[[ "$IMPL" == ref ]] && CTRL=(--controls)
run_logged parity "${TORCHRUN[@]}" $D12/check_pp.py --impl "$IMPL" --schedules gpipe,1f1b --microbatches "$MS" \
    --baseline "$BASE" --run-id "$RID" "${CTRL[@]}"

run_logged verify_core python - <<'PY'
import json, os
from pathlib import Path
out, rid, s, impl = Path(os.environ['DAY12_OUT']), os.environ['DAY12_RID'], Path(os.environ['DAY12_SESSION']), os.environ['DAY12_IMPL']
Ms = [int(m) for m in os.environ['DAY12_MS'].split(',')]
assert 'SIMULATOR CHECK: PASSED' in (s / 'sim_check.log').read_text()
assert 'DEADLOCK' in (s / 'sim_deadlock.log').read_text()
assert 'BOUNDARY DEMO: PASSED' in (s / 'boundary_demo.log').read_text()
b = json.loads((out / f'baseline_float32_{rid}.json').read_text())
assert b['repeat_bitwise'], 'baseline not reproducible on its own GPU'
assert any(l.startswith(f'BOUNDARY ({impl}') and l.rstrip().endswith('PASSED')
           for l in (s / 'boundary.log').read_text().splitlines()), 'boundary check did not pass'
assert json.loads((out / f'boundary_{impl}_{rid}.json').read_text())['passed']
r = json.loads((out / f'parity_{impl}_{rid}.json').read_text())
assert r['passed'] and r['exit_code'] == 0
for sch in ('gpipe', '1f1b'):
    for M in Ms:
        c = r['cases'][f'{sch}_M{M}']
        assert c['passed'], (sch, M, c['failed'])
        assert (out / f'events_{rid}' / f'{impl}_{sch}_M{M}.jsonl').is_file()
if impl == 'ref':
    assert len(r['controls']) == 3 and all(v['as_predicted'] for v in r['controls'].values())
print('OUTPUT CHECKS PASSED: simulator check, deadlock demo, boundary demo, baseline, boundary check, parity '
      f'GPipe and 1F1B at M={Ms}' + (', 3 controls detected as predicted' if impl == 'ref' else ''))
PY
printf 'Required stages and output checks passed.\n' > "$SESSION/CORE_SUCCESS.txt"
nvidia-smi --query-gpu=index,pstate,temperature.gpu,memory.used --format=csv | tee "$SESSION/gpu_after.txt"
df -h /workspace | tee "$SESSION/disk_after.txt"
printf 'All requested stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
