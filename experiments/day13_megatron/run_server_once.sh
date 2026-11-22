#!/usr/bin/env bash
# Day 13: every Day 13 stage once on a FOUR-GPU instance, checked and packed into a checksummed archive.
# Run from Linux after setup_megatron.sh; no driver installation and no automatic shutdown (stop the instance
# yourself after downloading).
#   bash experiments/day13_megatron/run_server_once.sh [run-id] [--impl ref|mine] [--mapper ref|mine] [--quick]
#                                                      [--no-sp] [--megatron PATH]
#   Stages: environment, topology (choose the fastest pair and the 4-GPU order), Megatron source stops
#           (find_symbols), rank mapper vs Megatron's RankGenerator (CPU), then GPU jobs:
#             tp1pp1    1 GPU   the reference losses (same full weights as every other layout)
#             tp2pp1    2 GPUs  smoke test + TP-only trace baseline, on the fastest-linked pair
#             tp1pp2    2 GPUs  smoke test + PP-only trace baseline, on the same pair
#             tp2pp2    4 GPUs  THE run: DP=1 TP=2 PP=2, sequence parallelism off
#             tp2pp2_sp 4 GPUs  optional comparison: the same with sequence parallelism on (skip with --no-sp)
#           then the trace analysis of each job and the cross-layout comparison.
#   --impl mine: forward_step_func/loss_func from my_step.py (YOU WRITE) instead of the reference.
#   --mapper mine: compare the built groups with your rank_mapper.my_groups instead of the reference mapper.
#   --quick: only tp1pp1 and tp2pp2.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=a100x4_$(date -u +%Y%m%dT%H%M%SZ)
IMPL=ref
MAPPER=ref
QUICK=0
SP=1
MEGATRON=/workspace/Megatron-LM
USAGE='Usage: run_server_once.sh [run-id] [--impl ref|mine] [--mapper ref|mine] [--quick] [--no-sp] [--megatron PATH]'
if [[ $# -gt 0 && "$1" != --* ]]; then RID=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --impl) IMPL=${2:-}; shift 2 ;;
        --mapper) MAPPER=${2:-}; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --no-sp) SP=0; shift ;;
        --megatron) MEGATRON=${2:-}; shift 2 ;;
        *) echo "$USAGE" >&2; exit 2 ;;
    esac
done
[[ "$IMPL" == ref || "$IMPL" == mine ]] || { echo '--impl must be ref or mine' >&2; exit 2; }
[[ "$MAPPER" == ref || "$MAPPER" == mine ]] || { echo '--mapper must be ref or mine' >&2; exit 2; }
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
[[ -d "$MEGATRON/megatron/core" ]] || { echo "No Megatron-LM at $MEGATRON: run setup_megatron.sh first" >&2; exit 2; }
OUT="$PROJECT/outputs/day13_megatron"
SESSION="$OUT/session_$RID"
RUN="$OUT/run_$RID"
mkdir -p "$OUT" "$PROJECT/outputs/day13_exports"
mkdir "$SESSION" # Refuse to overwrite an existing session.
ARCHIVE="$PROJECT/outputs/day13_exports/day13_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY13_SESSION="$SESSION" DAY13_OUT="$OUT" DAY13_RID="$RID" DAY13_MEGATRON="$MEGATRON"
printf 'run_id=%s\nimpl=%s\nmapper=%s\nquick=%s\nsp_stage=%s\nmegatron=%s\nstarted_utc=%s\n' "$RID" "$IMPL" \
    "$MAPPER" "$QUICK" "$SP" "$MEGATRON" "$(date -u +%FT%TZ)" > "$SESSION/commands.txt"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY13_OUT']), os.environ['DAY13_RID']
session = Path(os.environ['DAY13_SESSION'])
roots = [p for p in (session, out / f'run_{rid}') if p.exists()]
manifest = session / 'SHA256SUMS'
with manifest.open('w') as f:
    for root in roots:
        for p in sorted(root.rglob('*')):
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
    echo "Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
    echo "Run exit code: $rc (0 = all requested stages and output checks passed)"
    exit "$rc"
}
trap finish EXIT

source .venv/bin/activate
# CUDA_DEVICE_MAX_CONNECTIONS=1 must be in the environment BEFORE any process creates its CUDA context.
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$MEGATRON${PYTHONPATH:+:$PYTHONPATH}"
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO CUDA_VISIBLE_DEVICES

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
assert torch.cuda.is_available() and torch.cuda.device_count() >= 4, 'Expected at least four visible GPUs'
s, mg = Path(os.environ['DAY13_SESSION']), os.environ['DAY13_MEGATRON']
print('Platform:', platform.platform())
print('Python:', platform.python_version(), sys.executable)
print('Torch:', torch.__version__, '| CUDA build:', torch.version.cuda, '| NCCL:', '.'.join(map(str, torch.cuda.nccl.version())))
print('CONTAINER_ID:', os.environ.get('CONTAINER_ID', '?'), '| hostname:', platform.node())
print('CUDA_DEVICE_MAX_CONNECTIONS:', os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'))
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'cuda:{i}', p.name, f'{p.total_memory / 2**30:.1f} GiB', f'sm_{p.major}{p.minor}', f'{p.multi_processor_count} SMs')
n = torch.cuda.device_count()
print('P2P access matrix:', [[int(i == j or torch.cuda.can_device_access_peer(i, j)) for j in range(n)] for i in range(n)])
def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr
print(sh('nvidia-smi', '--query-gpu=index,name,driver_version,pstate,memory.used,memory.total,utilization.gpu', '--format=csv'))
print(sh('nvidia-smi', 'topo', '-m'))
print(sh('nvidia-smi', 'nvlink', '-s'))
print('Megatron-LM:', sh('git', '-C', mg, 'describe', '--tags').strip(), sh('git', '-C', mg, 'rev-parse', 'HEAD').strip())
import einops
print('einops:', einops.__version__)
snapshot = s / 'project_snapshot'
for folder in ['day13', 'experiments/day13_megatron']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
PY

D13=experiments/day13_megatron
run_logged topology python $D13/topo_check.py --out "$SESSION/gpu_choice.txt"
PAIR=$(sed -n 's/^pair=//p' "$SESSION/gpu_choice.txt")
ORDER=$(sed -n 's/^order=//p' "$SESSION/gpu_choice.txt")
printf 'pair=%s\norder=%s\n' "$PAIR" "$ORDER" >> "$SESSION/commands.txt"
run_logged symbols python $D13/find_symbols.py "$MEGATRON"
run_logged mapper python $D13/rank_mapper.py --impl "$MAPPER" --megatron "$MEGATRON" --layouts all \
    --json "$SESSION/mapper.json"

TRAIN=(python $D13/train_gpt.py --run-id "$RID" --impl "$IMPL" --mapper "$MAPPER")
torchrun_n() { python -m torch.distributed.run --standalone --nproc_per_node="$1" "${@:2}"; }
CUDA_VISIBLE_DEVICES=${PAIR%%,*} run_logged run_tp1pp1 torchrun_n 1 "${TRAIN[@]:1}" --tp 1 --pp 1
TAGS=(tp1pp1)
if [[ $QUICK == 0 ]]; then
    CUDA_VISIBLE_DEVICES=$PAIR run_logged run_tp2pp1 torchrun_n 2 "${TRAIN[@]:1}" --tp 2 --pp 1
    CUDA_VISIBLE_DEVICES=$PAIR run_logged run_tp1pp2 torchrun_n 2 "${TRAIN[@]:1}" --tp 1 --pp 2
    TAGS+=(tp2pp1 tp1pp2)
fi
CUDA_VISIBLE_DEVICES=$ORDER run_logged run_tp2pp2 torchrun_n 4 "${TRAIN[@]:1}" --tp 2 --pp 2
TAGS+=(tp2pp2)
if [[ $SP == 1 ]]; then
    CUDA_VISIBLE_DEVICES=$ORDER run_logged run_tp2pp2_sp torchrun_n 4 "${TRAIN[@]:1}" --tp 2 --pp 2 --sp
    TAGS+=(tp2pp2_sp)
fi
SUFFIX=$([[ $IMPL == mine ]] && echo _mine || true)
for t in "${TAGS[@]}"; do
    run_logged "analyze_$t" python $D13/analyze_trace.py "$RUN/$t$SUFFIX" --megatron "$MEGATRON"
done
run_logged compare python $D13/compare_runs.py "$RUN" --json "$SESSION/compare.json"

run_logged verify_core python - <<'PY'
import json, os
from pathlib import Path
s = Path(os.environ['DAY13_SESSION'])
assert 'RANK MAPPER' in (s / 'mapper.log').read_text() and 'PASSED' in (s / 'mapper.log').read_text().splitlines()[-1]
logs = sorted(s.glob('run_*.log'))
for log in logs:
    text = log.read_text()
    assert any(l.startswith('RUN ') and l.rstrip().endswith('PASSED') for l in text.splitlines()), f'{log.name} did not pass'
c = json.loads((s / 'compare.json').read_text())
assert c['passed'], 'compare_runs failed'
print(f'OUTPUT CHECKS PASSED: mapper vs RankGenerator, {len(logs)} Megatron jobs ({", ".join(l.stem[4:] for l in logs)}): '
      'groups == mapper, losses and step-0 gradient norms == 1-GPU reference, P2P shapes, NCCL counts, 1F1B order')
PY
printf 'Required stages and output checks passed.\n' > "$SESSION/CORE_SUCCESS.txt"
nvidia-smi --query-gpu=index,pstate,temperature.gpu,memory.used --format=csv | tee "$SESSION/gpu_after.txt"
df -h /workspace | tee "$SESSION/disk_after.txt"
printf 'All requested stages and output checks passed.\n' > "$SESSION/SUCCESS.txt"
