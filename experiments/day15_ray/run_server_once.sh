#!/usr/bin/env bash
# Day 15: every Day 15 stage once on a ONE-GPU instance (CPU rollout actors + 1 GPU learner), checked and packed
# into a checksummed archive. Run from Linux in the project root's venv; no driver installation and no automatic
# shutdown (stop - not destroy - the instance yourself after downloading).
#   bash experiments/day15_ray/run_server_once.sh [run-id] [--impl ref|mine] [--quick | --core]
#   Stages (all on one host; Ray starts and stops inside each Python stage):
#     environment      GPU, topology, NUMA, cgroup cpu.max/cpu.stat, /dev/shm, disk, versions, project snapshot
#     symbols          find_symbols.py on the installed ray package (Block 3 line numbers)
#     check_rl         GAE hand example + slow reference + 3 GAE controls; PPO surrogate unit tests   (CPU, no Ray)
#     probes           Exp 0: resources vs quota, placement env, call overhead, get vs wait, put once, zero copy
#     learn_*          learning sanity: CartPole, sync seeds 0-2 and async seed 0, 60 iterations, target 150
#     exp2_sync/async  Exp 2: CartPole, 2 actors + 1 GPU learner, phase breakdown, versions/digest/ratio checks
#     control_*        Exp 1 negative controls: skip_sync (digest check) and recompute_logp (ratio check)
#     trace_*          one traced short run per mode, ray.timeline() + read_timeline.py (never used for timing)
#     exp3_*           Exp 3: synthetic env, cost 0 / 400 us per step, 1/2/4 actors, sync and async
#     rllib            Exp 4: RLlib PPO with 2 EnvRunners + 1 GPU Learner, its timers + raw phase times
#     summarize        tables of Exp 2-4 into summary.md
#     verify_core      every required stage's final line checked
#   Optional (default mode only; failures are recorded, not fatal):
#     wait_n2/n4       ray.wait first-finished mode (IMPALA-style), lag histogram
#     repro_*          two identical runs of each mode: same final weight digest?
#     exp3_n8          8 actors with --ray-num-cpus 10: more actors than the cgroup quota
#     probes_oversub   4 actors x CPU inference at 1 vs 8 torch threads
#   --impl mine: check_rl, learn_* and exp2_* use my_rl.py (YOU WRITE: compute_gae, ppo_surrogate,
#                pipelined_iteration); controls, traces, exp3 and the optional stages use the reference.
#   --quick: environment, check_rl, learn_sync_s0, exp2 (10 iterations), the two controls.  --core: all required.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT=$PWD
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
IMPL=ref
MODE=full
USAGE='Usage: run_server_once.sh [run-id] [--impl ref|mine] [--quick | --core]'
if [[ $# -gt 0 && "$1" != --* ]]; then RID=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --impl) IMPL=${2:-}; shift 2 ;;
        --quick) MODE=quick; shift ;;
        --core) MODE=core; shift ;;
        *) echo "$USAGE" >&2; exit 2 ;;
    esac
done
[[ "$IMPL" == ref || "$IMPL" == mine ]] || { echo '--impl must be ref or mine' >&2; exit 2; }
[[ "$RID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run id'; exit 2; }
OUT="$PROJECT/outputs/day15_ray"
SESSION="$OUT/session_$RID"
RUN="$OUT/run_$RID"
mkdir -p "$OUT" "$RUN" "$PROJECT/outputs/day15_exports"
mkdir "$SESSION" # Refuse to overwrite an existing session.
ARCHIVE="$PROJECT/outputs/day15_exports/day15_$RID.tar.gz"
[[ ! -e "$ARCHIVE" ]] || { echo 'Archive already exists'; exit 2; }
export DAY15_SESSION="$SESSION" DAY15_OUT="$OUT" DAY15_RID="$RID" DAY15_MODE="$MODE" DAY15_IMPL="$IMPL"
printf 'run_id=%s\nimpl=%s\nmode=%s\nstarted_utc=%s\n' "$RID" "$IMPL" "$MODE" "$(date -u +%FT%TZ)" \
    > "$SESSION/commands.txt"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    ray stop --force > "$SESSION/ray_stop.txt" 2>&1
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$SESSION/gpu_processes_after.txt" 2>&1
    printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$SESSION/status.txt"
    python - <<'PY'
import hashlib, os
from pathlib import Path
out, rid = Path(os.environ['DAY15_OUT']), os.environ['DAY15_RID']
session = Path(os.environ['DAY15_SESSION'])
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
# Ray sets OMP_NUM_THREADS per worker from num_cpus when it is unset (GUIDE §1.2): leave it unset.
unset OMP_NUM_THREADS CUDA_VISIBLE_DEVICES RAY_ADDRESS
export PYTHONUNBUFFERED=1 RAY_DISABLE_DOCKER_CPU_WARNING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8

run_logged() {
    local name=$1
    shift
    printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$name"
    printf '%s: %s\n' "$name" "$*" >> "$SESSION/commands.txt"
    "$@" 2>&1 | tee "$SESSION/$name.log"
}
run_optional() {
    local name=$1 rc=0
    run_logged "$@" || rc=$?
    printf '%s exit_code=%s\n' "$name" "$rc" | tee -a "$SESSION/optional_status.txt"
}

df -h "$PROJECT" /dev/shm | tee "$SESSION/disk_before.txt"
run_logged environment python - <<'PY'
import json, os, platform, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, 'experiments/day15_ray')
from rl_common import host_info
s = Path(os.environ['DAY15_SESSION'])
info = host_info()
print(json.dumps(info, indent=1, default=str))
def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (r.stdout + r.stderr).strip()
    except OSError as e:
        return f'{cmd[0]}: {e}'
import torch
assert torch.cuda.is_available(), 'Expected a visible GPU'
p = torch.cuda.get_device_properties(0)
print('cuda:0', p.name, f'{p.total_memory / 2**30:.1f} GiB', f'sm_{p.major}{p.minor}', f'{p.multi_processor_count} SMs')
print(sh('nvidia-smi', '--query-gpu=index,name,driver_version,pstate,memory.used,memory.total,utilization.gpu,pci.bus_id', '--format=csv'))
print(sh('nvidia-smi', 'topo', '-m'))
print('numactl -H:\n' + sh('numactl', '-H'))
print('lscpu (model):', [l for l in sh('lscpu').splitlines() if l.startswith(('Model name', 'NUMA', 'Socket', 'Thread'))])
print('/dev/shm:', shutil.disk_usage('/dev/shm'))
print('pip:', sh(sys.executable, '-m', 'pip', 'list', '--format=freeze').replace('\n', ' ')[:4000])
snapshot = s / 'project_snapshot'
for folder in ['day15', 'experiments/day15_ray']:
    if Path(folder).is_dir():
        shutil.copytree(folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['README.md', 'requirements.txt']:
    if Path(name).is_file():
        shutil.copy2(name, snapshot / name)
(s / 'host.json').write_text(json.dumps(info, indent=1, default=str))
PY

D15=experiments/day15_ray
TRAIN=(python $D15/train.py)
run_logged symbols python $D15/find_symbols.py
run_logged check_rl python $D15/check_rl.py --impl "$IMPL" --json "$SESSION/check_rl.json"
if [[ $MODE != quick ]]; then
    RAY_DISABLE_DOCKER_CPU_WARNING=0 run_logged probes python $D15/ray_probes.py --json "$SESSION/probes.json"
fi
SEEDS=(0 1 2)
[[ $MODE == quick ]] && SEEDS=(0)
for s in "${SEEDS[@]}"; do
    run_logged learn_sync_s$s "${TRAIN[@]}" --impl "$IMPL" --mode sync --iters 60 --seed "$s" --learn-target 150 \
        --quiet --label learn --json "$SESSION/learn_sync_s$s.json"
done
if [[ $MODE != quick ]]; then
    run_logged learn_async_s0 "${TRAIN[@]}" --impl "$IMPL" --mode async --iters 60 --seed 0 --learn-target 150 \
        --quiet --label learn --json "$SESSION/learn_async_s0.json"
fi
ITERS=23
[[ $MODE == quick ]] && ITERS=10
for m in sync async; do
    run_logged exp2_$m "${TRAIN[@]}" --impl "$IMPL" --mode $m --iters $ITERS --warmup 3 --label exp2 \
        --json "$SESSION/exp2_$m.json"
done
for c in skip_sync recompute_logp; do
    run_logged control_$c "${TRAIN[@]}" --impl ref --mode async --control $c --iters 6 --warmup 1 --label control \
        --json "$SESSION/control_$c.json"
done
if [[ $MODE != quick ]]; then
    for m in sync async; do
        run_logged trace_$m bash -c "python $D15/train.py --mode $m --iters 6 --warmup 1 --quiet --label trace \
            --trace '$RUN/timeline_$m.json' && python $D15/read_timeline.py '$RUN/timeline_$m.json' --last 2"
    done
    for cost in 0 400; do
        for m in sync async; do
            for n in 1 2 4; do
                run_logged exp3_${m}_c${cost}_n$n "${TRAIN[@]}" --mode $m --env synthetic --obs-dim 16 \
                    --step-cost-us $cost --actors $n --frag-len $((4096 / (4 * n))) --minibatch 512 --iters 10 \
                    --warmup 2 --quiet --label exp3 --json "$SESSION/exp3_${m}_c${cost}_n$n.json"
            done
        done
    done
    run_logged rllib python $D15/rllib_ppo.py --iters 23 --warmup 3 --json "$SESSION/rllib.json"
    run_logged summarize python $D15/summarize.py "$SESSION"
fi

run_logged verify_core python - <<'PY'
import os
from pathlib import Path
s, mode = Path(os.environ['DAY15_SESSION']), os.environ['DAY15_MODE']
want = {'symbols': 'SYMBOLS: 0 not found', 'check_rl': 'CHECK_RL', 'learn_sync_s0': 'TRAIN', 'exp2_sync': 'TRAIN',
        'exp2_async': 'TRAIN', 'control_skip_sync': 'TRAIN', 'control_recompute_logp': 'TRAIN'}
if mode != 'quick':
    want |= {'probes': 'PROBES: done', 'learn_sync_s1': 'TRAIN', 'learn_sync_s2': 'TRAIN', 'learn_async_s0': 'TRAIN',
             'trace_sync': 'TIMELINE: done', 'trace_async': 'TIMELINE: done', 'rllib': 'RLLIB_PPO: done',
             'summarize': 'SUMMARY: done'}
    want |= {f'exp3_{m}_c{c}_n{n}': 'TRAIN' for m in ('sync', 'async') for c in (0, 400) for n in (1, 2, 4)}
for name, key in want.items():
    lines = [l.strip() for l in (s / f'{name}.log').read_text().splitlines() if l.startswith(key)]
    assert lines, f'{name}: no line starting with {key!r}'
    if key in ('CHECK_RL', 'TRAIN'):
        assert lines[-1].endswith('PASSED'), f'{name}: {lines[-1]}'
    print(f'{name:24s} {lines[-1][:150]}')
print(f'OUTPUT CHECKS PASSED ({mode}): {len(want)} stages')
PY
printf 'Required stages and output checks passed.\n' > "$SESSION/CORE_SUCCESS.txt"

if [[ $MODE == full ]]; then
    for n in 2 4; do
        run_optional wait_n$n "${TRAIN[@]}" --mode wait --actors $n --iters 40 --warmup 4 --quiet --label wait \
            --json "$SESSION/wait_n$n.json"
    done
    for m in sync async wait; do
        for k in 1 2; do
            run_optional repro_${m}_$k "${TRAIN[@]}" --mode $m --actors 2 --iters 12 --warmup 2 --quiet --label repro \
                --json "$SESSION/repro_${m}_$k.json"
        done
    done
    for m in sync async; do
        run_optional exp3_n8_$m "${TRAIN[@]}" --mode $m --env synthetic --obs-dim 16 --step-cost-us 400 --actors 8 \
            --frag-len 128 --minibatch 512 --iters 10 --warmup 2 --quiet --ray-num-cpus 10 --label exp3_n8 \
            --json "$SESSION/exp3n8_${m}.json"
    done
    run_optional probes_oversub python $D15/ray_probes.py --oversub --sizes-mib 1 --json "$SESSION/probes_oversub.json"
    run_optional summarize_full python $D15/summarize.py "$SESSION"
fi
nvidia-smi --query-gpu=index,pstate,temperature.gpu,memory.used --format=csv | tee "$SESSION/gpu_after.txt"
df -h "$PROJECT" /dev/shm | tee "$SESSION/disk_after.txt"
printf 'All requested stages ran; required output checks passed (optional exit codes: optional_status.txt).\n' \
    > "$SESSION/SUCCESS.txt"
