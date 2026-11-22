#!/usr/bin/env bash
# Day 13 follow-up: Nsight Systems instead of the PyTorch profiler (GUIDE Appendix C.6).
#  Part 1: the tp2pp2 job N times with the PyTorch profiler (fixed train_gpt.py) and N times with --profiler nsys,
#          interleaved; the nsys runs record only the timing step (cudaProfilerApi capture range).
#  Part 2: stall_probe.py for $LONG steps under nsys (whole run, CUDA API + OS runtime calls >= 10 ms), to see
#          whether the rare watchdog stall (C.3) shows up directly as a long cudaEventQuery and a long mutex wait.
# Usage, from /workspace/AI-Infra-Learning:  bash experiments/day13_megatron/run_nsys_check.sh <run-id> [N] [LONG]
set -uo pipefail
RID=${1:?run id}; N=${2:-3}; LONG=${3:-1200}
cd "$(dirname "$0")/../.."
source .venv/bin/activate >/dev/null
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3
D13=experiments/day13_megatron
S=outputs/day13_stall/nsys_$RID
mkdir -p "$S"
nsys --version > "$S/nsys_version.txt"
# no CPU sampling / context switches: perf_event_paranoid is 4 in this container
NSYS=(nsys profile -t cuda,nvtx,osrt --osrt-threshold=100000 --sample=none --cpuctxsw=none --cuda-event-trace=false -f true)

export_db() {   # export_db <nsys dir> <sqlite dir>
  for r in 0 1 2 3; do
    nsys export --type sqlite --force-overwrite true --output "$2/nsys_rank$r.sqlite" "$1/nsys_rank$r.nsys-rep" \
      >> "$S/export.log" 2>&1 || echo "EXPORT FAILED $1 rank $r"
  done
}

for i in $(seq 1 "$N"); do
  for v in torch nsys; do
    id=${RID}_${v}$i
    run=outputs/day13_megatron/run_$id
    echo "=== $id" | tee -a "$S/commands.txt"
    if [ $v = torch ]; then
      torchrun --standalone --nproc_per_node=4 $D13/train_gpt.py --tp 2 --pp 2 --run-id "$id" > "$S/$id.log" 2>&1 \
        || echo "FAILED $id"
    else
      mkdir -p "$run/nsys"
      torchrun --standalone --nproc_per_node=4 --no-python \
        "${NSYS[@]}" --capture-range=cudaProfilerApi --capture-range-end=stop -o "$run/nsys/nsys_rank%q{RANK}" \
        python -u $D13/train_gpt.py --tp 2 --pp 2 --profiler nsys --run-id "$id" > "$S/$id.log" 2>&1 \
        || echo "FAILED $id"
      export_db "$run/nsys" "$run/tp2pp2"
    fi
    python $D13/analyze_trace.py "$run/tp2pp2" --megatron /workspace/Megatron-LM > "$S/analyze_$id.log" 2>&1 \
      || echo "ANALYZE FAILED $id"
  done
done

python - "$S" "$RID" "$N" <<'PY'
import json, sys
from pathlib import Path
s, rid, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = []
for i in range(1, n + 1):
    for v in ("torch", "nsys"):
        run = Path(f"outputs/day13_megatron/run_{rid}_{v}{i}/tp2pp2")
        r0 = json.loads((run / "rank0.json").read_text())
        ta = json.loads((run / "trace_analysis.json").read_text())["timing"]
        st = {x["step"]: round(x["ms"], 1) for x in r0["steps"]}
        key = lambda kv: int(kv[0])  # noqa: E731
        idle = {r: round(t["idle_other_ms"], 2) for r, t in sorted(ta.items(), key=key)}
        win = {r: round(t["window_ms"], 1) for r, t in sorted(ta.items(), key=key)}
        gap = {r: round(max((g["len_ms"] for g in t["longest_idle_gaps"]), default=0), 2) for r, t in sorted(ta.items(), key=key)}
        order = {r: " ".join(t["order"]) for r, t in sorted(ta.items(), key=key)}
        rows.append({"run": f"{v}{i}", "step_ms": st, "window_ms": win, "idle_other_ms": idle, "longest_gap_ms": gap,
                     "order": order})
        print(f"{v}{i}: step ms {st}; timing window {win}; idle other/CPU {idle}; longest gap {gap}")
print("orders identical across modes:", len({json.dumps(r["order"], sort_keys=True) for r in rows}) == 1)
(s / "nsys_check.json").write_text(json.dumps(rows, indent=1))
PY

# ---- Part 2: the long probe run under nsys ------------------------------------------------------------------
if [ "$LONG" -gt 0 ]; then
  P=outputs/day13_stall/nsys_${RID}_probe
  mkdir -p "$P"
  echo "=== probe $LONG steps under nsys" | tee -a "$S/commands.txt"
  torchrun --standalone --nproc_per_node=4 --no-python \
    nsys profile -t cuda,osrt --osrt-threshold=10000000 --sample=none --cpuctxsw=none --cuda-event-trace=false -f true \
    -o "$P/nsys_rank%q{RANK}" python -u $D13/stall_probe.py --run-id "nsys_${RID}" --steps "$LONG" \
    > "$S/probe.log" 2>&1 || echo "FAILED probe"
  export_db "$P" "$P"
  python - "$P" "$S" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "experiments/day13_megatron")
import nsys_trace
p, s = Path(sys.argv[1]), Path(sys.argv[2])
lines = []
for r in range(4):
    v = nsys_trace.thread_view(p / f"nsys_rank{r}.sqlite", min_ms=20, top=6)
    lines.append(f"=== rank {r}: calls of 20 ms or more over the whole run (at_ms = from the first CUDA call)")
    lines.append(nsys_trace.summarize_threads(v, top_threads=8))
txt = "\n".join(lines)
print(txt)
(s / "probe_threads.txt").write_text(txt)
PY
  python $D13/nsys_watchdog.py "$P" > /dev/null && cp "$P/watchdog_report.txt" "$S/"
  sed -n '/^STALL PROBE/,$p' "$S/probe.log" | grep -v "^\[rank\|Warning" | head -30 > "$S/probe_report.txt"
  cat "$S/probe_report.txt"
fi
echo "NSYS CHECK $RID done"
