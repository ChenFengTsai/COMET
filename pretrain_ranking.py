import json

path = "/storage/ssd1/richtsai1103/vid2act/log/metaworld/mt6/10_top50/coffee_push/original_metrics_seed456/metrics.jsonl"

# 1) Find contrib keys present in the file (distill/contrib_0, distill/contrib_1, ...)
all_keys = set()
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            all_keys.update(json.loads(line).keys())

contrib_keys = sorted(
    [k for k in all_keys if k.startswith("distill/contrib_")],
    key=lambda x: int(x.split("_")[-1])
)

if not contrib_keys:
    raise RuntimeError("No distill/contrib_* keys found in the file.")

# 2) Find the last log line that contains any distill/contrib_* key
last_idx = None
last_obj = None

with open(path, "r", encoding="utf-8") as f:
    rows = [line for line in f if line.strip()]

for i in range(len(rows) - 1, -1, -1):
    obj = json.loads(rows[i])
    if any(k in obj for k in contrib_keys):
        last_idx = i
        last_obj = obj
        break

if last_obj is None:
    raise RuntimeError("No log line contains distill/contrib_* keys (unexpected).")

# 3) Rank contribs in that last log entry (descending by value)
contrib = {k: last_obj[k] for k in contrib_keys if k in last_obj}
ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)

# "last contrib" interpreted as the highest index contrib key, e.g. contrib_5
last_contrib_key = contrib_keys[-1]
rank_of_last_contrib = next(
    (rank for rank, (k, _) in enumerate(ranked, start=1) if k == last_contrib_key),
    None
)

print(f"Last contrib-kind log is at 0-based line index: {last_idx} (1-based: {last_idx + 1})")
print(f"step: {last_obj.get('step')}")
print("\nRanking (desc):")
for r, (k, v) in enumerate(ranked, start=1):
    print(f"{r}. {k} = {v}")

print(f"\nRank of last contrib ({last_contrib_key}): {rank_of_last_contrib} / {len(ranked)}")
