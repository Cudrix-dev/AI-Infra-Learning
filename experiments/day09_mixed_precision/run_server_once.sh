#!/usr/bin/env bash
# Day 09: run every Day 09 stage once on the dual A100, check the outputs, and pack a checksummed archive.
# Run from Linux; no driver installation and no automatic shutdown. Same structure as Days 04, 06, 07 and 08.
#   bash experiments/day09_mixed_precision/run_server_once.sh [run-id] [--core|--full]
#   --core: the required stages only (dtype probe, fp16 order, checkpoint probe, the 6 required matrix rows)
#   --full (default): also the optional matrix rows, fp16 --underflow and the two-GPU DDP question
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
OUT="$PROJECT/outputs/day09_amp"
SESSION="$OUT/session_$RID"
mkdir -p "$OUT"
mkdir "$SESSION" # Refuse to overwrite an existing session.
mkdir -p "$PROJECT/outputs/day09_exports"
ARCHIVE="$PROJECT/outputs/day09_exports/day09_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY09_SESSION="$SESSION" DAY09_OUT="$OUT" DAY09_RID="$RID" DAY09_MODE="$MODE"
printf 'run_id=%s\nmode=%s\nstarted_utc=%s\n' "$RID" "$MODE" "$(date -u +%FT%TZ)" > "$SESSION/commands.txt"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY09_OUT']), os.environ['DAY09_RID']
session = Path(os.environ['DAY09_SESSION'])
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
    echo "Run exit code: $rc (0 = all requested stages and output checks passed)"
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
    printf '%s: %s\n' "$name" "$*" >> "$SESSION/commands.txt"
    "$@" 2>&1 | tee "$SESSION/$name.log"
}

df -h /workspace | tee "$SESSION/disk_before.txt"
run_logged environment python - <<'PY'
import os, platform, shutil, subprocess, torch
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Need two visible CUDA GPUs'
s = Path(os.environ['DAY09_SESSION'])
print('Platform:', platform.platform())
print('Python:', platform.python_version())
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version(),
      'NCCL:', torch.cuda.nccl.version())
print('bf16 supported:', torch.cuda.is_bf16_supported(), '| TF32 matmul default:', torch.backends.cuda.matmul.allow_tf32)
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
print('CONTAINER_ID:', os.environ.get('CONTAINER_ID', '?'), '| hostname:', platform.node())
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, f'{p.total_memory / 2**30:.1f} GiB', f'sm_{p.major}{p.minor}')
print(subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True,
                     text=True).stdout)
print(subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True).stdout)
snapshot = s / 'project_snapshot'
for folder in ['day09', 'experiments/day01_single_gpu', 'experiments/day02_data_parallel',
               'experiments/day09_mixed_precision']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
src = s / 'installed_torch_source'
root = Path(torch.__file__).resolve().parent
for rel in ['amp/autocast_mode.py', 'amp/grad_scaler.py', 'nn/utils/clip_grad.py', 'utils/checkpoint.py',
            'distributed/algorithms/ddp_comm_hooks/default_hooks.py']:
    (src / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rel, src / rel)
PY
run_logged probe_before python experiments/day02_data_parallel/gpu_probe.py --want 2
run_logged find_symbols python experiments/day09_mixed_precision/find_symbols.py

D9=experiments/day09_mixed_precision
run_logged dtypes python $D9/probe_amp.py --part dtypes --run-id "$RID"
if [[ "$MODE" == --full ]]; then
    run_logged fp16_order python $D9/fp16_order.py --underflow --run-id "$RID"
else
    run_logged fp16_order python $D9/fp16_order.py --run-id "$RID"
fi
run_logged ckpt python $D9/probe_amp.py --part ckpt --run-id "$RID"
if [[ "$MODE" == --full ]]; then
    run_logged matrix python $D9/compare_amp.py --stage matrix --condition shared --run-id "$RID"
else
    run_logged matrix python $D9/compare_amp.py --stage matrix --condition shared --run-id "$RID" --core
fi

verify_outputs() {
    run_logged "$1" python - "$2" <<'PY'
import json, os, sys
from pathlib import Path
out, rid, s = Path(os.environ['DAY09_OUT']), os.environ['DAY09_RID'], Path(os.environ['DAY09_SESSION'])
full = sys.argv[1] == '--full'  # also check the DDP stage (runs after the core check)
matrix_full = os.environ['DAY09_MODE'] == '--full'  # the matrix ran its optional rows too
assert 'params, grads and AdamW state fp32 under fp32/bf16/fp16 autocast: True' in (s / 'dtypes.log').read_text()
assert 'FP16 ORDER: ALL CHECKS PASSED' in (s / 'fp16_order.log').read_text()
assert 'CKPT CHECK: PASSED' in (s / 'ckpt.log').read_text()
mx = json.loads((out / f'matrix_summary_{rid}.json').read_text())
want = 9 if matrix_full else 6
assert len(mx['table']) == want, (len(mx['table']), want)
assert mx['ckpt_pass'], 'checkpointing changed the numbers'
for name in ('probe_dtypes_ref', 'probe_ckpt_ref', 'fp16_order_ref'):
    json.loads((out / f'{name}_{rid}.json').read_text())
dirs = [p for p in out.iterdir() if p.is_dir() and p.name.endswith('_' + rid) and p != s]
runs = [p for p in dirs if p.name.startswith('mx_')]
assert len(runs) == want * mx['repeats'], (len(runs), want * mx['repeats'])
for p in dirs:
    json.loads((p / 'metrics.json').read_text())
    assert not (p / 'params_final.pt').exists() and not (p / 'grads0.pt').exists(), f'large tensor left in {p}'
n_ddp = 0
if full:
    d = json.loads((out / f'ddp_summary_{rid}.json').read_text())
    assert any('BFloat16' in k for k in d['ddp_bf16hook']['collectives']), d['ddp_bf16hook']['collectives']
    n_ddp = len([p for p in dirs if p.name.startswith('ddp_')])
    assert n_ddp == 3, n_ddp
print(f'OUTPUT CHECKS PASSED: dtype probe, fp16 order (control detected), checkpoint probe, {len(runs)} matrix runs'
      + (f', {n_ddp} DDP runs' if full else ''))
PY
}

verify_outputs verify_core --core-check
printf 'Required dtype, fp16-order, checkpoint and matrix stages passed.\n' > "$SESSION/CORE_SUCCESS.txt"
if [[ "$MODE" == --full ]]; then
    run_logged ddp python $D9/compare_amp.py --stage ddp --condition shared --run-id "$RID"
    verify_outputs verify_outputs --full
fi
run_logged probe_after python experiments/day02_data_parallel/gpu_probe.py --want 2
df -h /workspace | tee "$SESSION/disk_after.txt"
printf 'All requested stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
