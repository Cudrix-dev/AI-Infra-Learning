#!/usr/bin/env bash
# Day 10: run every Day 10 stage once on ONE GPU, check the outputs, and pack a checksummed archive.
# Run from Linux; no driver installation and no automatic shutdown (stop the instance yourself after downloading).
#   bash experiments/day10_triton_softmax/run_server_once.sh [run-id] [--impl ref|mine] [--quick|--core|--full]
#   --quick: checks on 13 row lengths + the quick benchmark (~30 s; for trying your kernel)
#   --core (default): full correctness sweep (+ negative controls for ref), kernel inspection, benchmark stages
#                     rows/count/padding/warps
#   --full: --core + the optional launch stage (torch.compile reduce-overhead) + the RMSNorm stretch check
#   GPU=1 bash ... selects the second GPU (default GPU=0).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
IMPL=ref
MODE=--core
if [[ $# -gt 0 && "$1" != --* ]]; then RID=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --impl) IMPL=${2:-}; shift 2 ;;
        --quick|--core|--full) MODE=$1; shift ;;
        *) echo 'Usage: run_server_once.sh [run-id] [--impl ref|mine] [--quick|--core|--full]' >&2; exit 2 ;;
    esac
done
[[ "$IMPL" == ref || "$IMPL" == mine ]] || { echo '--impl must be ref or mine' >&2; exit 2; }
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
OUT="$PROJECT/outputs/day10_softmax"
SESSION="$OUT/session_$RID"
mkdir -p "$OUT" "$PROJECT/outputs/day10_exports"
mkdir "$SESSION" # Refuse to overwrite an existing session.
ARCHIVE="$PROJECT/outputs/day10_exports/day10_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY10_SESSION="$SESSION" DAY10_OUT="$OUT" DAY10_RID="$RID" DAY10_MODE="$MODE" DAY10_IMPL="$IMPL"
printf 'run_id=%s\nimpl=%s\nmode=%s\nstarted_utc=%s\n' "$RID" "$IMPL" "$MODE" "$(date -u +%FT%TZ)" > "$SESSION/commands.txt"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY10_OUT']), os.environ['DAY10_RID']
session = Path(os.environ['DAY10_SESSION'])
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
# One GPU. Fresh compiler caches, so first-call times include cold JIT/Inductor compilation (not archived).
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/day10_cache_$RID/triton TORCHINDUCTOR_CACHE_DIR=/tmp/day10_cache_$RID/inductor
printf 'CUDA_VISIBLE_DEVICES=%s TRITON_CACHE_DIR=%s TORCHINDUCTOR_CACHE_DIR=%s\n' "$CUDA_VISIBLE_DEVICES" \
    "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" >> "$SESSION/commands.txt"

run_logged() {
    local name=$1
    shift
    printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$name"
    printf '%s: %s\n' "$name" "$*" >> "$SESSION/commands.txt"
    "$@" 2>&1 | tee "$SESSION/$name.log"
}

df -h /workspace | tee "$SESSION/disk_before.txt"
run_logged environment python - <<'PY'
import os, platform, shutil, subprocess, sys, torch, triton
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected exactly one visible GPU'
s = Path(os.environ['DAY10_SESSION'])
print('Platform:', platform.platform())
print('Python:', platform.python_version(), sys.executable)
print('Torch:', torch.__version__, '| CUDA build:', torch.version.cuda, '| cuDNN:', torch.backends.cudnn.version())
print('Triton:', triton.__version__)
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
print('CONTAINER_ID:', os.environ.get('CONTAINER_ID', '?'), '| hostname:', platform.node())
p = torch.cuda.get_device_properties(0)
print(p.name, f'{p.total_memory / 2**30:.1f} GiB', f'sm_{p.major}{p.minor}', f'{p.multi_processor_count} SMs',
      f'L2 {p.L2_cache_size / 2**20:.0f} MiB')
def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr
print(sh('nvidia-smi', '--query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,'
         'power.limit,temperature.gpu,memory.used', '--format=csv'))
print(sh('nvidia-smi', 'topo', '-m'))
(s / 'nvidia_smi_q.txt').write_text(sh('nvidia-smi', '-q', '-d', 'CLOCK,PERFORMANCE,POWER'))
snapshot = s / 'project_snapshot'
for folder in ['day10', 'experiments/day10_triton_softmax']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
src = s / 'installed_source'
troot, xroot = Path(torch.__file__).resolve().parent, Path(triton.__file__).resolve().parent
for root, rel in [(xroot, 'language/standard.py'), (xroot, 'runtime/jit.py'),
                  (troot, '_decomp/decompositions.py'), (troot, '_inductor/runtime/triton_heuristics.py')]:
    (src / root.name / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rel, src / root.name / rel)
PY

D10=experiments/day10_triton_softmax
run_logged find_symbols python $D10/find_symbols.py
CHECK_ARGS=(--impl "$IMPL" --run-id "$RID")
[[ "$IMPL" == ref ]] && CHECK_ARGS+=(--controls)
[[ "$MODE" == --quick ]] && CHECK_ARGS+=(--quick)
run_logged check python $D10/check_softmax.py "${CHECK_ARGS[@]}"
if [[ "$MODE" == --quick ]]; then
    run_logged bench python $D10/bench_softmax.py --impl "$IMPL" --stage quick --run-id "$RID"
else
    run_logged inspect python $D10/inspect_kernels.py --impl "$IMPL" --run-id "$RID"
    run_logged bench python $D10/bench_softmax.py --impl "$IMPL" --stage core --run-id "$RID"
fi

run_logged verify_core python - <<'PY'
import json, os
from pathlib import Path
out, rid, s = Path(os.environ['DAY10_OUT']), os.environ['DAY10_RID'], Path(os.environ['DAY10_SESSION'])
impl, mode = os.environ['DAY10_IMPL'], os.environ['DAY10_MODE']
log = (s / 'check.log').read_text()
assert 'SOFTMAX CHECK: PASSED' in log, 'correctness check did not pass'
if impl == 'ref':
    assert 'CONTROLS: ALL DETECTED' in log, 'a negative control was not detected'
chk = json.loads((out / f'check_softmax_{impl}_{rid}.json').read_text())
assert chk['passed'] and not chk['contract_problems']
stage = 'quick' if mode == '--quick' else 'core'
bench = json.loads((out / f'bench_{impl}_{stage}_{rid}.json').read_text())
assert (out / f'bench_{impl}_{stage}_{rid}.csv').is_file()
want = {'quick': 12, 'core': 20 * 5 + 12 * 5 + 4 * 4 + 16}[stage]
assert len(bench['rows']) == want, (len(bench['rows']), want)
bad = [r for r in bench['rows'] if r['close_to_torch'] is False and r['impl'] != 'naive']
assert not bad, bad[:3]
if mode != '--quick':
    assert (out / f'inspect_{impl}_{rid}' / 'inspect.json').is_file()
print(f"OUTPUT CHECKS PASSED: {len(chk['cases'])} correctness cases, {len(bench['rows'])} timed cases ({stage})"
      + (', controls detected' if impl == 'ref' else ''))
PY
printf 'Required check, inspection and benchmark stages passed.\n' > "$SESSION/CORE_SUCCESS.txt"

if [[ "$MODE" == --full ]]; then
    run_logged bench_launch python $D10/bench_softmax.py --impl "$IMPL" --stage launch --run-id "$RID"
    set +e
    python $D10/check_softmax.py --impl "$IMPL" --op rmsnorm --run-id "$RID" 2>&1 | tee "$SESSION/check_rmsnorm.log"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -eq 3 ]]; then echo 'RMSNorm stretch not implemented for this impl: skipped.'
    elif [[ $rc -ne 0 ]]; then echo 'RMSNorm check failed'; exit 1; fi
    run_logged verify_full python - <<'PY'
import json, os
from pathlib import Path
out, rid, impl = Path(os.environ['DAY10_OUT']), os.environ['DAY10_RID'], os.environ['DAY10_IMPL']
b = json.loads((out / f'bench_{impl}_launch_{rid}.json').read_text())
assert len(b['rows']) == 24, len(b['rows'])
assert any(r['impl'] == 'compile_ro' for r in b['rows'])
print('OUTPUT CHECKS PASSED: launch stage (24 timed cases)')
PY
fi
nvidia-smi --query-gpu=index,pstate,clocks.sm,clocks.mem,temperature.gpu,memory.used --format=csv | tee "$SESSION/gpu_after.txt"
df -h /workspace | tee "$SESSION/disk_after.txt"
rm -rf "/tmp/day10_cache_$RID"
printf 'All requested stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
