#!/usr/bin/env bash
# Run from Linux; no driver/dependency installation and no automatic shutdown.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=${1:-a100_$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
OUT="$PROJECT/outputs/day04_grad_accum"
SESSION="$OUT/session_$RID"
mkdir -p "$OUT"
mkdir "$SESSION" # Refuse to overwrite an existing session.
mkdir -p "$PROJECT/outputs/day04_exports"
ARCHIVE="$PROJECT/outputs/day04_exports/day04_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY04_SESSION="$SESSION" DAY04_OUT="$OUT" DAY04_RID="$RID"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    # Include only this run, including partial results if a stage failed.
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY04_OUT']), os.environ['DAY04_RID']
session = Path(os.environ['DAY04_SESSION'])
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

run_logged environment python - <<'PY'
import os, platform, shutil, torch
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Need two visible CUDA GPUs'
s = Path(os.environ['DAY04_SESSION'])
print('Platform:', platform.platform())
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'NCCL:', torch.cuda.nccl.version())
print('Torch git revision:', torch.version.git_version)
print('Python:', platform.python_version())
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
for i in range(2):
    print(i, torch.cuda.get_device_name(i))
snapshot = s / 'project_snapshot'
for folder in ['day04', 'experiments/day01_single_gpu', 'experiments/day02_data_parallel',
               'experiments/day03_ddp_hooks', 'experiments/day04_grad_accum']:
    shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    shutil.copy2(name, snapshot / name)
src = s / 'installed_torch_source'
src.mkdir()
torch_root = Path(torch.__file__).resolve().parent
for rel in ['nn/parallel/distributed.py', '_tensor.py']:
    target = src / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(torch_root / rel, target)
# Best effort: C++ sources are usually absent from the wheel. Keep failures visible.
import urllib.request
rev = torch.version.git_version
for rel in ['torch/csrc/distributed/c10d/reducer.cpp', 'torch/csrc/distributed/c10d/reducer.hpp',
            'torch/csrc/autograd/functions/accumulate_grad.h']:
    url = f'https://raw.githubusercontent.com/pytorch/pytorch/{rev}/{rel}'
    print('C++ source:', url, flush=True)
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = response.read()
        (src / Path(rel).name).write_bytes(data)
    except Exception as exc:
        print('OPTIONAL SOURCE DOWNLOAD FAILED:', exc)
PY
run_logged probe_before python experiments/day02_data_parallel/gpu_probe.py --want 2
run_logged find_symbols python experiments/day04_grad_accum/find_symbols.py
run_logged semantics python experiments/day04_grad_accum/accum_semantics.py --device cuda

# Plain python: compare_accum launches its own one-/two-process workers.
run_logged parity python experiments/day04_grad_accum/compare_accum.py --stage parity --condition shared --run-id "$RID"
for stage in count timing slowlink; do
    run_logged "$stage" python experiments/day04_grad_accum/compare_accum.py \
        --stage "$stage" --condition shared --run-id "$RID" --no-smoke --repeats 2
done
run_logged probe_after python experiments/day02_data_parallel/gpu_probe.py --want 2
run_logged verify_outputs python - <<'PY'
import json, os
from pathlib import Path
out, rid = Path(os.environ['DAY04_OUT']), os.environ['DAY04_RID']
s = Path(os.environ['DAY04_SESSION'])
assert 'PARITY: ALL CHECKS PASSED' in (s / 'parity.log').read_text()
assert (s / 'parity.log').read_text().count('=> DETECTED by check') == 5
for stage, expected in [('count', 7), ('timing', 32), ('slowlink', 8)]:
    rows = json.loads((out / f'{stage}_summary_{rid}.json').read_text())
    assert len(rows) == expected, (stage, len(rows))
    if stage == 'count':
        assert all(row['allreduce_per_step'] > 0 for row in rows), 'No NCCL kernels detected; inspect traces'
dirs = [p for p in out.iterdir() if p.is_dir() and p.name.endswith('_' + rid) and p != s]
assert len(dirs) == 103, f'Expected 103 variant runs, found {len(dirs)}'
for p in dirs:
    json.loads((p / 'metrics.json').read_text())
    assert (p / 'result.pt').is_file(), p
traces = [t for p in dirs for t in p.glob('trace_rank*.json')]
assert len(traces) == 14, f'Expected 14 traces, found {len(traces)}'
print('OUTPUT CHECKS PASSED: 103 runs, 14 traces, 3 summary tables, 5 detected controls')
PY
printf 'All stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
