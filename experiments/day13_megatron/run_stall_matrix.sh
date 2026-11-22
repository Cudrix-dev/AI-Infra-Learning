#!/usr/bin/env bash
# Day 13 follow-up: reproduce the host stall, isolate its cause, and test fixes (GUIDE Appendix B).
# Usage, from /workspace/AI-Infra-Learning:  bash experiments/day13_megatron/run_stall_matrix.sh <run-id>
# About 6 minutes on four A100s. Writes outputs/day13_stall/session_<run-id>/ (one log per case) and
# outputs/day13_stall/run_<run-id>_<case>/ (per-rank JSON, summary.json, report.txt).
set -uo pipefail
RID=${1:?run id}
cd "$(dirname "$0")/../.."
source .venv/bin/activate >/dev/null
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
D13=experiments/day13_megatron
S=outputs/day13_stall/session_$RID
mkdir -p "$S"
PROF="10:stack,11:timing,20:stack,21:timing,30:stack,31:timing"   # the train_gpt.py pattern, three times

run() {   # run <case> <args...>
  local case=$1; shift
  echo "=== $case: $*" | tee -a "$S/commands.txt"
  torchrun --standalone --nproc_per_node=4 $D13/stall_probe.py --run-id "${RID}_$case" "$@" > "$S/$case.log" 2>&1
  local rc=$?
  sed -n '/^STALL PROBE/,$p' "$S/$case.log" | grep -v "^\[rank\|Warning" | head -40
  echo "exit $rc"
  return $rc
}

fail=0
# 1. Is there a stall without the profiler at all? (300 steps, default GC)
run base_unprofiled --steps 300 || fail=1
# 2. Reproduce: the profiler pattern of train_gpt.py, default GC, profiler released "later" (as train_gpt.py did)
run prof_later --steps 40 --profile-steps "$PROF" --gc-debug-steps 10,11,12 || fail=1
# 3. Fix A: free the profiler explicitly, outside the timed step
run prof_collect --steps 40 --profile-steps "$PROF" --release-profiler collect || fail=1
# 4. Fix B: Megatron-style manual GC (disable + freeze, collect every 10 steps between steps)
run prof_manual_gc --steps 40 --profile-steps "$PROF" --gc manual --gc-interval 10 || fail=1
# 5. Fix B must not cost anything in steady state
run manual_gc_unprofiled --steps 300 --gc manual --gc-interval 10 || fail=1
# 6. CPU placement: bind each rank to its GPU's NUMA CPUs (steady state)
run pin_numa_unprofiled --steps 300 --pin-numa || fail=1

echo "exit_code=$fail" > "$S/status.txt"
echo "STALL MATRIX $RID: exit $fail"
