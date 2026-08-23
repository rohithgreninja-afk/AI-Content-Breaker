import os, time, csv, random
import requests

BASE = "https://datasets-server.huggingface.co/rows"
DS, CFG, SPLIT = "liamdugan/raid", "raid", "train"
OUT_DIR = r"C:\Projects\AI-Content-Breaker\data"
OUT = os.path.join(OUT_DIR, "raid_clean.csv")
os.makedirs(OUT_DIR, exist_ok=True)

N_ROWS = 2200000
HUMAN_CAP = 1200
AI_PER_MODEL_CAP = 180
AI_TOTAL_CAP = 1300
MAX_REQUESTS = 300
BLOCK = 50
random.seed(42)

sess = requests.Session()

def fetch(offset):
    for attempt in range(5):
        r = sess.get(BASE, params={"dataset": DS, "config": CFG, "split": SPLIT,
                                   "offset": offset, "length": BLOCK}, timeout=60)
        if r.status_code == 200:
            return r.json().get("rows", [])
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (attempt + 1))
            continue
        return []
    return []

seen = set()
human, ai_total = 0, 0
per_model = {}
rows_out = []

def done():
    return human >= HUMAN_CAP and ai_total >= AI_TOTAL_CAP

reqs = 0
while reqs < MAX_REQUESTS and not done():
    off = random.randint(0, N_ROWS - BLOCK)
    reqs += 1
    for item in fetch(off):
        row = item.get("row", {})
        if row.get("attack") != "none":
            continue
        rid = row.get("id")
        if rid in seen:
            continue
        model = row.get("model")
        gen = (row.get("generation") or "").strip()
        if len(gen) < 40:
            continue
        if model == "human":
            if human >= HUMAN_CAP:
                continue
            human += 1
            label = "human"
        else:
            if ai_total >= AI_TOTAL_CAP or per_model.get(model, 0) >= AI_PER_MODEL_CAP:
                continue
            per_model[model] = per_model.get(model, 0) + 1
            ai_total += 1
            label = "ai"
        seen.add(rid)
        rows_out.append([rid, model, label, row.get("domain"), gen])
    if reqs % 20 == 0:
        print(f"req {reqs}: human={human} ai={ai_total} models={per_model}", flush=True)
    time.sleep(0.25)

random.shuffle(rows_out)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "model", "label", "domain", "text"])
    w.writerows(rows_out)

print("\nDONE", flush=True)
print(f"requests used: {reqs}", flush=True)
print(f"total rows: {len(rows_out)} | human={human} ai={ai_total}", flush=True)
print(f"AI per model: {per_model}", flush=True)
print(f"saved -> {OUT}", flush=True)
