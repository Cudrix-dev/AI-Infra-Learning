"""Day 01 - where does the GPU time of one medium step go? (math vs fused attention)

Run each mode in its own process (use_deterministic_algorithms is process-global):
  python experiments/day01_single_gpu/profile_attention.py --mode det
  python experiments/day01_single_gpu/profile_attention.py --mode fast
Each prints the top GPU kernels and writes a Chrome trace you can open at https://ui.perfetto.dev
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from train import OUTPUT_ROOT, PRESETS, Config, build, get_batch, setup_reproducibility, train_step

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["det", "fast"], required=True)
p.add_argument("--steps", type=int, default=5)
args = p.parse_args()

cfg = Config(**PRESETS["medium"])
cfg.deterministic = args.mode == "det"
setup_reproducibility(cfg)
device = torch.device("cuda:0")
model, optimizer, _ = build(cfg, device)

for s in range(5):  # warmup: cuBLAS handles, workspaces, lazy Adam state, allocator growth
    train_step(model, optimizer, *get_batch(cfg, s, device), cfg)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for s in range(5, 5 + args.steps):
        with record_function(f"step{s}"):
            train_step(model, optimizer, *get_batch(cfg, s, device), cfg)
    torch.cuda.synchronize()

# GPU time per kernel name, summed over the profiled steps. Attention-related rows to look for:
# det : bmm/gemm on [B*H,T,T], softmax + _softmax_backward_data, add (mask), mul (scaling), copies
# fast: efficient_attention / fmha / cudnn kernels (forward + backward)
table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=30, max_name_column_width=70)
print(f"=== mode={args.mode}, {args.steps} steps (divide totals by {args.steps} for per-step) ===")
print(table)
out = OUTPUT_ROOT / f"trace_{args.mode}.json"
out.parent.mkdir(parents=True, exist_ok=True)
prof.export_chrome_trace(str(out))
print(f"trace: {out}")
