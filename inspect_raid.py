from datasets import load_dataset
from collections import Counter

print("Loading RAID in streaming mode (no full download)...", flush=True)
dd = load_dataset("liamdugan/raid", streaming=True)
print("SPLITS:", list(dd.keys()), flush=True)

split = list(dd.keys())[0]
ds = dd[split]
it = iter(ds)

first = next(it)
print("\nCOLUMNS:", list(first.keys()), flush=True)

print("\n--- FIRST ROW (values truncated) ---", flush=True)
for k, v in first.items():
    s = str(v)
    if len(s) > 160:
        s = s[:160] + "..."
    print(f"{k}: {s}", flush=True)

print("\n--- tallying model/attack/domain over first 400 rows ---", flush=True)
mc, ac, dc = Counter(), Counter(), Counter()
mc[first.get("model")] += 1
ac[first.get("attack")] += 1
dc[first.get("domain")] += 1
for i, row in enumerate(it):
    if i >= 399:
        break
    mc[row.get("model")] += 1
    ac[row.get("attack")] += 1
    dc[row.get("domain")] += 1

print("MODEL counts:", dict(mc), flush=True)
print("ATTACK counts:", dict(ac), flush=True)
print("DOMAIN counts:", dict(dc), flush=True)
