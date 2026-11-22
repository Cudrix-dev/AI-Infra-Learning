#!/usr/bin/env bash
# Day 13 follow-up: does the train_gpt.py fix remove the timing-step stall? (GUIDE Appendix B)
# Runs the tp2pp2 job N times with the old file (train_gpt_before.py, a copy of the pre-fix train_gpt.py) and
# N times with the fixed train_gpt.py, analyzes every trace, and prints the timing step per run.
# Usage, from /workspace/AI-Infra-Learning:  bash experiments/day13_megatron/run_stall_fix_check.sh <run-id> [N]
set -uo pipefail
RID=${1:?run id}; N=${2:-3}
cd "$(dirname "$0")/../.."
source .venv/bin/activate >/dev/null
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM OMP_NUM_THREADS=1
D13=experiments/day13_megatron
S=outputs/day13_stall/fixcheck_$RID
mkdir -p "$S"
for i in $(seq 1 "$N"); do
  for v in before after; do
    f=$D13/train_gpt.py; [ $v = before ] && f=$D13/train_gpt_before.py
    id=${RID}_${v}$i
    echo "=== $id: $f" | tee -a "$S/commands.txt"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 $f --tp 2 --pp 2 --run-id "$id" \
      > "$S/$id.log" 2>&1 || echo "FAILED $id"
    python $D13/analyze_trace.py outputs/day13_megatron/run_$id/tp2pp2 --megatron /workspace/Megatron-LM \
      > "$S/analyze_$id.log" 2>&1 || echo "ANALYZE FAILED $id"
  done
done
python - "$S" "$RID" "$N" <<'PY'
import json, sys
from pathlib import Path
s, rid, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = []
for i in range(1, n + 1):
    for v in ("before", "after"):
        run = Path(f"outputs/day13_megatron/run_{rid}_{v}{i}/tp2pp2")
        r0 = json.loads((run / "rank0.json").read_text())
        ta = json.loads((run / "trace_analysis.json").read_text())["timing"]
        st = {x["step"]: round(x["ms"], 1) for x in r0["steps"]}
        idle = {r: round(t["idle_other_ms"], 2) for r, t in sorted(ta.items(), key=lambda kv: int(kv[0]))}
        gap = {r: max((g["len_ms"] for g in t["longest_idle_gaps"]), default=0) for r, t in ta.items()}
        rows.append({"run": f"{v}{i}", "step_ms": st, "timing_step_idle_other_ms": idle,
                     "timing_step_longest_gap_ms": {r: round(g, 2) for r, g in sorted(gap.items(), key=lambda kv: int(kv[0]))}})
        print(f"{v}{i}: step ms {st}; timing step idle(other/CPU) per rank {idle}; longest gap {rows[-1]['timing_step_longest_gap_ms']}")
(s / "fixcheck.json").write_text(json.dumps(rows, indent=1))
PY
echo "FIX CHECK $RID done"
