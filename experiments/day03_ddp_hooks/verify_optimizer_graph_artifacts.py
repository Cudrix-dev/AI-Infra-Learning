"""Check saved medium-run loss curves, counters and LR schedules without more GPU work."""
import json
import math
from pathlib import Path
import sys

import torch
from train_ddp import Config, PRESETS, lr_lambda


def main():
    root = Path(sys.argv[1])
    checked = []
    for repeat in range(3):
        for impl in ("mini_ddp", "torch_ddp"):
            results = [torch.load(root / f"timing_r{repeat}_{impl}_{mode}" / "result.pt",
                                  map_location="cpu", weights_only=False)
                       for mode in ("ready_eager", "graph")]
            assert results[0]["losses"] == results[1]["losses"], (repeat, impl)
            for r in results:
                cfg = Config(**PRESETS["medium"])
                cfg.total_steps = 60
                assert set(r["meta"]["adam_steps"]) == {60.0}
                assert r["meta"]["scheduler_last_epoch"] == 60
                assert all(math.isclose(v, cfg.lr * lr_lambda(i, cfg), rel_tol=2e-6)
                           for i, v in enumerate(r["meta"]["lr_used"]))
            checked.append({"repeat": repeat, "impl": impl, "loss_curve_bitwise": True,
                            "adam_and_lr_checked": True})
    result = {"all_passed": True, "medium_checks": checked}
    (root / "validation_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
