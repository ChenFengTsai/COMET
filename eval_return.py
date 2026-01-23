import json
import numpy as np
from pathlib import Path

BASE_DIR = Path("/storage/ssd1/richtsai1103/vid2act/log/metaworld/mt6/10_top50/coffee_push")
SEEDS = [0, 123, 456, 789, 2024]

def extract_final_mean_eval_return(metrics_path: Path) -> float:
    vals = []
    with metrics_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "mean_eval_return" in rec and isinstance(rec["mean_eval_return"], (int, float)):
                vals.append(float(rec["mean_eval_return"]))
    if not vals:
        raise ValueError(f"No 'mean_eval_return' found in {metrics_path}")
    return vals[-1]  # final logged eval return

finals = []
per_seed = {}

for seed in SEEDS:
    metrics_path = BASE_DIR / f"original_seed{seed}" / "metrics.jsonl"
    v = extract_final_mean_eval_return(metrics_path)
    per_seed[seed] = v
    finals.append(v)
    print(f"seed={seed}: final mean_eval_return = {v:.6f}")

finals = np.array(finals, dtype=float)
mean_ = finals.mean()
sd_ = finals.std(ddof=1) if len(finals) > 1 else float("nan")

print("\n=== Aggregated across seeds ===")
print(f"mean = {mean_:.6f}")
print(f"sd   = {sd_:.6f}  (sample SD, ddof=1)")
