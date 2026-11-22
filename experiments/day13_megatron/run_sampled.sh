#!/usr/bin/env bash
# Day 13 follow-up: 600 unprofiled steps with host_sampler.py and nvidia-smi running alongside (GUIDE Appendix B).
set -uo pipefail
RID=${1:?run id}; STEPS=${2:-600}
cd "$(dirname "$0")/../.."
source .venv/bin/activate >/dev/null
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM OMP_NUM_THREADS=1
D13=experiments/day13_megatron
RUN=outputs/day13_stall/run_$RID      # written by stall_probe.py (must not exist yet)
OUT=outputs/day13_stall/host_$RID     # sampler, nvidia-smi and probe log
PIDS=outputs/day13_stall/pids_$RID
rm -rf "$PIDS"; mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 50 > "$OUT/smi.csv" 2>&1 &
SMI=$!
if [ "${CATCH:-0}" = 1 ]; then   # CATCH=1: also take gdb stacks in the first stall of each rank
  python $D13/stall_catcher.py "$PIDS" "$OUT/catch" > "$OUT/catcher.out" 2>&1 &
  EXTRA="--allow-ptrace"
fi
python $D13/host_sampler.py "$PIDS" "$OUT/host.jsonl" --period-ms 5 > "$OUT/sampler.log" 2>&1 &
SAMP=$!
torchrun --standalone --nproc_per_node=4 $D13/stall_probe.py --run-id "$RID" --steps "$STEPS" --pid-dir "$PIDS" ${EXTRA:-} \
  > "$OUT/probe.log" 2>&1
rc=$?
wait $SAMP; kill $SMI 2>/dev/null; wait
python $D13/stall_correlate.py "$RUN" "$OUT/host.jsonl" --smi "$OUT/smi.csv" > /dev/null 2>"$OUT/correlate.err"
echo "probe exit $rc"; sed -n '/^STALL PROBE/,$p' "$OUT/probe.log" | grep -v "^\[rank\|Warning" | head -25
