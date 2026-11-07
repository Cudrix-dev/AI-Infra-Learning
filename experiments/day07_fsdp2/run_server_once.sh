#!/usr/bin/env bash
# Day 07: run the whole FSDP2 matrix once on the dual A100, check the outputs, and pack a checksummed archive.
# Run from Linux; no driver/dependency installation and no automatic shutdown. Same structure as Days 04 and 06.
#   bash experiments/day07_fsdp2/run_server_once.sh [run-id]
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=${1:-a100_$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
OUT="$PROJECT/outputs/day07_fsdp2"
SESSION="$OUT/session_$RID"
mkdir -p "$OUT"
mkdir "$SESSION" # Refuse to overwrite an existing session.
mkdir -p "$PROJECT/outputs/day07_exports"
ARCHIVE="$PROJECT/outputs/day07_exports/day07_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY07_SESSION="$SESSION" DAY07_OUT="$OUT" DAY07_RID="$RID"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    # Include only this run (also partial results if a stage failed). Traces are large: they are packed too, because
    # analyze_fsdp_trace.py re-reads them locally; the archive is then a few hundred MB.
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY07_OUT']), os.environ['DAY07_RID']
session = Path(os.environ['DAY07_SESSION'])
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
import os, platform, shutil, subprocess, torch
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Need two visible CUDA GPUs'
s = Path(os.environ['DAY07_SESSION'])
print('Platform:', platform.platform())
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'NCCL:', torch.cuda.nccl.version())
print('Torch git revision:', torch.version.git_version)
print('Python:', platform.python_version())
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
for i in range(2):
    print(i, torch.cuda.get_device_name(i))
print(subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True).stdout)
snapshot = s / 'project_snapshot'
for folder in ['day07', 'experiments/day01_single_gpu', 'experiments/day02_data_parallel',
               'experiments/day03_ddp_hooks', 'experiments/day04_grad_accum', 'experiments/day07_fsdp2']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
src = s / 'installed_torch_source'
torch_root = Path(torch.__file__).resolve().parent
shutil.copytree(torch_root / 'distributed/fsdp/_fully_shard', src / 'distributed/fsdp/_fully_shard',
                ignore=shutil.ignore_patterns('__pycache__'))
PY
run_logged probe_before python experiments/day02_data_parallel/gpu_probe.py --want 2
run_logged find_symbols python experiments/day07_fsdp2/find_symbols.py

# Plain python: compare_fsdp launches its own one-/two-process workers.
run_logged parity python experiments/day07_fsdp2/compare_fsdp.py --stage parity --condition shared --run-id "$RID"
run_logged count python experiments/day07_fsdp2/compare_fsdp.py --stage count --condition shared --run-id "$RID" \
    --no-smoke
run_logged memory python experiments/day07_fsdp2/compare_fsdp.py --stage memory --condition shared --run-id "$RID" \
    --no-smoke
run_logged timing python experiments/day07_fsdp2/compare_fsdp.py --stage timing --condition shared --run-id "$RID" \
    --no-smoke --repeats 3
run_logged slowlink python experiments/day07_fsdp2/compare_fsdp.py --stage slowlink --condition shared \
    --run-id "$RID" --no-smoke --repeats 2
run_logged probe_after python experiments/day02_data_parallel/gpu_probe.py --want 2
run_logged verify_outputs python - <<'PY'
import json, os
from pathlib import Path
out, rid = Path(os.environ['DAY07_OUT']), os.environ['DAY07_RID']
s = Path(os.environ['DAY07_SESSION'])
log = (s / 'parity.log').read_text()
assert 'PARITY: ALL CHECKS PASSED' in log
assert log.count('=> DETECTED by check') == 3
assert len(json.loads((out / f'timing_summary_{rid}.json').read_text())) == 18
assert len(json.loads((out / f'slowlink_summary_{rid}.json').read_text())['timing']) == 5
assert len(json.loads((out / f'memory_summary_{rid}.json').read_text())) == 5
cnt = json.loads((out / f'count_summary_{rid}.json').read_text())
assert cnt['block_sync_k4_m8']['per_step'].get('AllGather') == 52, cnt['block_sync_k4_m8']
dirs = [p for p in out.iterdir() if p.is_dir() and p.name.endswith('_' + rid) and p != s]
expected_runs = 13 + 15 + 5 + 18 * 3 + 5 * 2 + 4
assert len(dirs) == expected_runs, f'Expected {expected_runs} runs, found {len(dirs)}'
for p in dirs:
    json.loads((p / 'metrics.json').read_text())
    assert (p / 'result.pt').is_file(), p
traces = [t for p in dirs for t in p.glob('trace_rank*.json')]
assert len(traces) == 38, f'Expected 38 traces, found {len(traces)}'
snaps = [t for p in dirs for t in p.glob('memory_snapshot_rank0.pickle')]
assert len(snaps) == 5, f'Expected 5 memory snapshots, found {len(snaps)}'
print(f'OUTPUT CHECKS PASSED: {expected_runs} runs, 38 traces, 5 memory snapshots, 4 summary tables, '
      f'3 detected controls')
PY
df -h /workspace | tee "$SESSION/disk_after.txt"
printf 'All stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
