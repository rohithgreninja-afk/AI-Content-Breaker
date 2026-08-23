import requests

BASE = "https://datasets-server.huggingface.co"
DS = "liamdugan/raid"

r = requests.get(f"{BASE}/splits", params={"dataset": DS}, timeout=60)
print("SPLITS:", r.json(), flush=True)

r = requests.get(f"{BASE}/size", params={"dataset": DS}, timeout=60)
try:
    sz = r.json()
    print("SIZE keys:", list(sz.keys()), flush=True)
    print("SIZE:", sz.get("size", {}).get("splits") or sz, flush=True)
except Exception as e:
    print("size err", e, r.text[:300], flush=True)

def peek(offset, cfg="raid", split="train", length=4):
    r = requests.get(f"{BASE}/rows", params={
        "dataset": DS, "config": cfg, "split": split,
        "offset": offset, "length": length}, timeout=60)
    if r.status_code != 200:
        print(f"offset {offset}: HTTP {r.status_code} {r.text[:200]}", flush=True)
        return
    rows = r.json().get("rows", [])
    for item in rows:
        row = item["row"]
        print(f"  off{offset}: model={row.get('model')} attack={row.get('attack')} domain={row.get('domain')}", flush=True)

print("\n--- peeking at spread-out offsets ---", flush=True)
for off in [0, 100000, 500000, 1500000, 3000000, 5000000]:
    peek(off)
