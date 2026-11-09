#!/usr/bin/env bash
# Day 08: run the whole DeepSpeed ZeRO matrix once on the dual A100, check the outputs, and pack a checksummed archive.
# Run from Linux; no driver installation and no automatic shutdown. Same structure as Days 04, 06 and 07.
#   bash experiments/day08_deepspeed/run_server_once.sh [run-id] [--core]
# DeepSpeed must already be installed in the venv (guide "Read this first": pip install deepspeed==0.19.9).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=${1:-a100_$(date -u +%Y%m%dT%H%M%SZ)}
MODE=${2:---full}
if [[ $# -gt 2 || ( "$MODE" != --full && "$MODE" != --core ) ]]; then
    echo 'Usage: run_server_once.sh [run-id] [--core|--full]' >&2
    exit 2
fi
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
OUT="$PROJECT/outputs/day08_deepspeed"
SESSION="$OUT/session_$RID"
mkdir -p "$OUT"
mkdir "$SESSION" # Refuse to overwrite an existing session.
mkdir -p "$PROJECT/outputs/day08_exports"
ARCHIVE="$PROJECT/outputs/day08_exports/day08_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY08_SESSION="$SESSION" DAY08_OUT="$OUT" DAY08_RID="$RID"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY08_OUT']), os.environ['DAY08_RID']
session = Path(os.environ['DAY08_SESSION'])
roots = sorted(p for p in out.iterdir() if p.name.endswith('_' + rid) or p.name.endswith('_' + rid + '.json'))
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
    local manifest_rc=$?
    if [[ $manifest_rc -ne 0 ]]; then
        echo "Manifest failed; download $OUT manually."
        exit 1
    fi
    tar -czf "$ARCHIVE" -C "$OUT" -T "$SESSION/archive_members.txt"
    local pack_rc=$?
    if [[ $pack_rc -eq 0 ]]; then
        (cd "$(dirname "$ARCHIVE")" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
        pack_rc=$?
    fi
    if [[ $pack_rc -ne 0 ]]; then
        echo "Packaging failed; download $OUT manually."
        exit 1
    fi
    echo "Archive: $ARCHIVE"
    echo "Run exit code: $rc (0 = all stages and output checks passed)"
    exit "$rc"
}
trap finish EXIT

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO

run_logged() {
    local name=$1
    shift
    printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$name"
    "$@" 2>&1 | tee "$SESSION/$name.log"
}

df -h /workspace | tee "$SESSION/disk_before.txt"
run_logged environment python - <<'PY'
import os, platform, shutil, subprocess, torch, deepspeed
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Need two visible CUDA GPUs'
s = Path(os.environ['DAY08_SESSION'])
print('Platform:', platform.platform())
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'NCCL:', torch.cuda.nccl.version())
print('DeepSpeed:', deepspeed.__version__, 'at', Path(deepspeed.__file__).parent)
print('Python:', platform.python_version())
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
for i in range(2):
    print(i, torch.cuda.get_device_name(i))
print(subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True).stdout)
snapshot = s / 'project_snapshot'
for folder in ['day08', 'experiments/day01_single_gpu', 'experiments/day02_data_parallel',
               'experiments/day03_ddp_hooks', 'experiments/day04_grad_accum', 'experiments/day07_fsdp2',
               'experiments/day08_deepspeed']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
src = s / 'installed_deepspeed_source'
ds_root = Path(deepspeed.__file__).resolve().parent
for rel in ['runtime/engine.py', 'runtime/zero', 'runtime/comm/coalesced_collectives.py', 'runtime/utils.py',
            'utils/tensor_fragment.py']:
    p = ds_root / rel
    if p.is_dir():
        shutil.copytree(p, src / rel, ignore=shutil.ignore_patterns('__pycache__'))
    else:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, src / rel)
PY
run_logged probe_before python experiments/day02_data_parallel/gpu_probe.py --want 2
run_logged find_symbols python experiments/day08_deepspeed/find_symbols.py

# Plain python: compare_ds launches its own one-/two-process workers.
run_logged parity python experiments/day08_deepspeed/compare_ds.py --stage parity --condition shared --run-id "$RID"
run_logged count python experiments/day08_deepspeed/compare_ds.py --stage count --condition shared --run-id "$RID" \
    --no-smoke
run_logged memory python experiments/day08_deepspeed/compare_ds.py --stage memory --condition shared --run-id "$RID" \
    --no-smoke
verify_outputs() {
    run_logged "$1" python - "$2" <<'PY'
import json, os
import sys
from pathlib import Path
out, rid = Path(os.environ['DAY08_OUT']), os.environ['DAY08_RID']
s = Path(os.environ['DAY08_SESSION'])
log = (s / 'parity.log').read_text()
assert 'PARITY: ALL CHECKS PASSED' in log
assert log.count('=> DETECTED by check') == 2
full = sys.argv[1] == '--full'
if full:
    assert len(json.loads((out / f'timing_summary_{rid}.json').read_text())) == 19
    assert len(json.loads((out / f'slowlink_summary_{rid}.json').read_text())['timing']) == 5
assert len(json.loads((out / f'memory_summary_{rid}.json').read_text())) == 8
cnt = json.loads((out / f'count_summary_{rid}.json').read_text())
assert len([k for k in cnt if not k.endswith('_sequence')]) == 11, sorted(cnt)
dirs = [p for p in out.iterdir() if p.is_dir() and p.name.endswith('_' + rid) and p != s]
expected_runs = 8 + 11 + 8 + (19 * 3 + 5 * 2 + 2 if full else 0)
assert len(dirs) == expected_runs, f'Expected {expected_runs} runs, found {len(dirs)}'
for p in dirs:
    json.loads((p / 'metrics.json').read_text())
    assert (p / 'result.pt').is_file(), p
traces = [t for p in dirs for t in p.glob('trace_rank*.json')]
expected_traces = 22 + (4 if full else 0)
assert len(traces) == expected_traces, f'Expected {expected_traces} traces, found {len(traces)}'
snaps = [t for p in dirs for t in p.glob('memory_snapshot_rank0.pickle')]
assert len(snaps) == 8, f'Expected 8 memory snapshots, found {len(snaps)}'
print(f'OUTPUT CHECKS PASSED: {expected_runs} runs, {expected_traces} traces, 8 memory snapshots, '
      f'{4 if full else 2} JSON summaries, 2 detected controls')
PY
}

verify_outputs verify_core --core
printf 'Required parity, count and memory stages passed.\n' > "$SESSION/CORE_SUCCESS.txt"
if [[ "$MODE" == --full ]]; then
    run_logged timing python experiments/day08_deepspeed/compare_ds.py --stage timing --condition shared \
        --run-id "$RID" --no-smoke --repeats 3
    run_logged slowlink python experiments/day08_deepspeed/compare_ds.py --stage slowlink --condition shared \
        --run-id "$RID" --no-smoke --repeats 2
    verify_outputs verify_outputs --full
fi
run_logged probe_after python experiments/day02_data_parallel/gpu_probe.py --want 2
df -h /workspace | tee "$SESSION/disk_after.txt"
printf 'All requested stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
