#!/usr/bin/env bash
# Day 14: every Day 14 stage once on a FOUR-GPU instance, checked and packed into a checksummed archive.
# Run from Linux in the project root's venv; no driver installation and no automatic shutdown (stop - not
# destroy - the instance yourself after downloading).
#   bash experiments/day14_moe/run_server_once.sh [run-id] [--impl ref|mine] [--quick | --core] [--megatron PATH]
#   Stages (one GPU = cuda:0 of the same host; EP=4 = all four GPUs, one process each):
#     environment      GPUs, P2P matrix, nvidia-smi topo -m, versions, project snapshot
#     symbols          find_symbols.py on the Megatron checkout (skipped if there is none; never fatal)
#     check_local      hand-worked examples + CPU-simulated EP=4/EP=8 dispatch          (CPU)
#     route_stats      Exp 1: routing, capacity, receive rows on the medium preset         (1 GPU)
#     baseline         the one-GPU oracle: EP=1, all experts, every routing pattern       (1 GPU)
#     ep4_dispatch     exchange_counts + labelled-row all_to_all forward/backward          (4 GPUs)
#     ep4_parity       Exp 2: EP=4 vs the one-GPU baseline, 4 routing cases                (4 GPUs)
#     ep4_controls     Exp 2: the five negative controls, each must fail as predicted      (4 GPUs)
#     bench_ep1/ep4    Exp 3: skew timing, one GPU then EP=4                                (1 / 4 GPUs)
#     verify_core      every required stage's final line checked
#   Optional (default mode only; failures are recorded, not fatal):
#     ep2_parity       EP=2 x EDP=2 vs the same baseline (expert grads all-reduced over EDP)
#     aux_local        EP=4 with Megatron's default LOCAL aux scope: expected to differ from the baseline (P9)
#     aux_train        route_stats --aux-train 200 (a hypothesis about load recovery)
#     bench_trace      one profiled step per traced case, after all timed steps
#   --impl mine: check_local, baseline and the EP=4 dispatch/parity stages, plus the optional ep2_parity and
#                aux_local stages, use my_moe.py (YOU WRITE); parity is checked against the REFERENCE
#                baseline; controls, route_stats and timing always use the reference.
#   --quick: environment, check_local, baseline, ep4_dispatch, ep4_parity only.  --core: all required stages.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=a100x4_$(date -u +%Y%m%dT%H%M%SZ)
IMPL=ref
MODE=full
MEGATRON=/workspace/Megatron-LM
USAGE='Usage: run_server_once.sh [run-id] [--impl ref|mine] [--quick | --core] [--megatron PATH]'
if [[ $# -gt 0 && "$1" != --* ]]; then RID=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --impl) IMPL=${2:-}; shift 2 ;;
        --quick) MODE=quick; shift ;;
        --core) MODE=core; shift ;;
        --megatron) MEGATRON=${2:-}; shift 2 ;;
        *) echo "$USAGE" >&2; exit 2 ;;
    esac
done
[[ "$IMPL" == ref || "$IMPL" == mine ]] || { echo '--impl must be ref or mine' >&2; exit 2; }
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
OUT="$PROJECT/outputs/day14_moe"
SESSION="$OUT/session_$RID"
RUN="$OUT/run_$RID"
mkdir -p "$OUT" "$RUN" "$PROJECT/outputs/day14_exports"
mkdir "$SESSION" # Refuse to overwrite an existing session.
ARCHIVE="$PROJECT/outputs/day14_exports/day14_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY14_SESSION="$SESSION" DAY14_OUT="$OUT" DAY14_RID="$RID" DAY14_MEGATRON="$MEGATRON" DAY14_MODE="$MODE" \
    DAY14_IMPL="$IMPL"
printf 'run_id=%s\nimpl=%s\nmode=%s\nmegatron=%s\nstarted_utc=%s\n' "$RID" "$IMPL" "$MODE" "$MEGATRON" \
    "$(date -u +%FT%TZ)" > "$SESSION/commands.txt"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY14_OUT']), os.environ['DAY14_RID']
session = Path(os.environ['DAY14_SESSION'])
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

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO CUDA_VISIBLE_DEVICES

run_logged() {
    local name=$1
    shift
    printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$name"
    printf '%s: %s\n' "$name" "$*" >> "$SESSION/commands.txt"
    "$@" 2>&1 | tee "$SESSION/$name.log"
}
# optional stages: record the exit code, never stop the run
run_optional() {
    local name=$1 rc=0
    run_logged "$@" || rc=$?
    printf '%s exit_code=%s\n' "$name" "$rc" | tee -a "$SESSION/optional_status.txt"
}

df -h "$PROJECT" | tee "$SESSION/disk_before.txt"
run_logged environment python - <<'PY'
import os, platform, shutil, subprocess, sys, torch
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.device_count() >= 4, 'Expected at least four visible GPUs'
s, mg = Path(os.environ['DAY14_SESSION']), os.environ['DAY14_MEGATRON']
print('Platform:', platform.platform())
print('Python:', platform.python_version(), sys.executable)
print('Torch:', torch.__version__, '| CUDA build:', torch.version.cuda, '| NCCL:', '.'.join(map(str, torch.cuda.nccl.version())))
print('CONTAINER_ID:', os.environ.get('CONTAINER_ID', '?'), '| hostname:', platform.node())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'cuda:{i}', p.name, f'{p.total_memory / 2**30:.1f} GiB', f'sm_{p.major}{p.minor}', f'{p.multi_processor_count} SMs')
n = torch.cuda.device_count()
print('P2P access matrix:', [[int(i == j or torch.cuda.can_device_access_peer(i, j)) for j in range(n)] for i in range(n)])
def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout + r.stderr
    except OSError as e:
        return f'{cmd[0]}: {e}'
print(sh('nvidia-smi', '--query-gpu=index,name,driver_version,pstate,memory.used,memory.total,utilization.gpu', '--format=csv'))
print(sh('nvidia-smi', 'topo', '-m'))
if Path(mg, 'megatron/core').is_dir():
    print('Megatron-LM:', sh('git', '-C', mg, 'describe', '--tags').strip(), sh('git', '-C', mg, 'rev-parse', 'HEAD').strip())
else:
    print('Megatron-LM: not found at', mg, '(symbols stage skipped)')
snapshot = s / 'project_snapshot'
for folder in ['day14', 'experiments/day14_moe']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
PY

D14=experiments/day14_moe
if [[ -d "$MEGATRON/megatron/core/transformer/moe" ]]; then
    run_optional symbols python $D14/find_symbols.py "$MEGATRON"
fi
run_logged check_local python $D14/check_local.py --impl "$IMPL" --json "$SESSION/check_local.json"
if [[ $MODE != quick ]]; then
    CUDA_VISIBLE_DEVICES=0 run_logged route_stats python $D14/route_stats.py --json "$SESSION/route_stats.json"
fi
BASE="$RUN/baseline_small_ref.pt"
CUDA_VISIBLE_DEVICES=0 run_logged baseline python $D14/baseline_1gpu.py --run-id "$RID" --impl ref --out "$BASE"
if [[ $IMPL == mine ]]; then
    CUDA_VISIBLE_DEVICES=0 run_logged baseline_mine python $D14/baseline_1gpu.py --run-id "$RID" --impl mine \
        --out "$RUN/baseline_small_mine.pt"
fi
EP4=(env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4)
run_logged ep4_dispatch "${EP4[@]}" $D14/check_ep.py --impl "$IMPL" --tests dispatch \
    --json "$SESSION/ep4_dispatch.json"
run_logged ep4_parity "${EP4[@]}" $D14/check_ep.py --impl "$IMPL" --baseline "$BASE" \
    --json "$SESSION/ep4_parity.json"
if [[ $MODE != quick ]]; then
    run_logged ep4_controls "${EP4[@]}" $D14/check_ep.py --impl controls --baseline "$BASE" \
        --json "$SESSION/ep4_controls.json"
    CUDA_VISIBLE_DEVICES=0 run_logged bench_ep1 python $D14/bench_skew.py --ep 1 --json "$SESSION/bench_ep1.json"
    run_logged bench_ep4 "${EP4[@]}" $D14/bench_skew.py --ep 4 --json "$SESSION/bench_ep4.json"
fi

run_logged verify_core python - <<'PY'
import os
from pathlib import Path
s, mode = Path(os.environ['DAY14_SESSION']), os.environ['DAY14_MODE']
want = {'check_local': 'CHECK_LOCAL', 'baseline': 'BASELINE written', 'ep4_dispatch': 'DISPATCH', 'ep4_parity': 'PARITY'}
if mode != 'quick':
    want |= {'route_stats': 'ROUTE_STATS: done', 'ep4_controls': 'CONTROLS', 'bench_ep1': 'BENCH_SKEW: done',
             'bench_ep4': 'BENCH_SKEW: done'}
for name, key in want.items():
    lines = [l.strip() for l in (s / f'{name}.log').read_text().splitlines() if l.startswith(key)]
    assert lines, f'{name}: no line starting with {key!r}'
    if key in ('CHECK_LOCAL', 'DISPATCH', 'PARITY', 'CONTROLS'):
        assert lines[-1].endswith('PASSED'), f'{name}: {lines[-1]}'
    print(f'{name:13s} {lines[-1]}')
print(f'OUTPUT CHECKS PASSED ({mode}): ' + ', '.join(want))
PY
printf 'Required stages and output checks passed.\n' > "$SESSION/CORE_SUCCESS.txt"

if [[ $MODE == full ]]; then
    run_optional ep2_parity "${EP4[@]}" $D14/check_ep.py --impl "$IMPL" --ep 2 --baseline "$BASE" \
        --json "$SESSION/ep2_parity.json"
    run_optional aux_local "${EP4[@]}" $D14/check_ep.py --impl "$IMPL" --aux-scope local \
        --cases learned,balanced,ragged --baseline "$BASE" --json "$SESSION/aux_local.json"
    CUDA_VISIBLE_DEVICES=0 run_optional aux_train python $D14/route_stats.py --aux-train 200 \
        --json "$SESSION/aux_train.json"
    run_optional bench_trace "${EP4[@]}" $D14/bench_skew.py --ep 4 --shapes large --patterns balanced,all_to_one \
        --steps 5 --trace "$RUN/traces" --json "$SESSION/bench_trace.json"
fi
nvidia-smi --query-gpu=index,pstate,temperature.gpu,memory.used --format=csv | tee "$SESSION/gpu_after.txt"
df -h "$PROJECT" | tee "$SESSION/disk_after.txt"
printf 'All requested stages ran; required output checks passed (optional exit codes: optional_status.txt).\n' \
    > "$SESSION/SUCCESS.txt"
