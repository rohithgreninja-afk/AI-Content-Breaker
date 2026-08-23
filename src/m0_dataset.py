"""M0: build a balanced clean RAID subset at data/raid_clean.csv.

Streams liamdugan/raid from the HF hub (robust; avoids the throttled
datasets-server rows API), shuffles a buffer for class/model spread, keeps only
clean rows (attack == "none") with usable text, and collects a small balanced
subset. Caps are MVP-sized so it finishes fast; raise them to scale up later.

Frozen CSV contract: id, model, label, domain, text  (label in {human, ai}).
"""

from __future__ import annotations

import csv
import os
from collections import Counter

from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "data", "raid_clean.csv")

HUMAN_CAP = 400
AI_TOTAL_CAP = 400
AI_PER_MODEL_CAP = 60
MIN_TEXT_LEN = 40
SHUFFLE_BUFFER = 10000
MAX_SCAN = 200000
SEED = 42


def get_stream():
    """Return a shuffled streaming iterator over the RAID train split."""
    try:
        stream = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)
    except Exception:
        splits = load_dataset("liamdugan/raid", streaming=True)
        stream = splits[list(splits.keys())[0]]
    return stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)


def is_clean(row: dict) -> bool:
    """True only for unattacked rows with a long-enough generation."""
    if row.get("attack") != "none":
        return False
    return len((row.get("generation") or "").strip()) >= MIN_TEXT_LEN


def collect() -> list:
    """Stream and collect a balanced clean subset as CSV-ready rows."""
    human = 0
    ai_total = 0
    per_model: dict[str, int] = {}
    rows_out: list = []
    scanned = 0

    for row in get_stream():
        scanned += 1
        if scanned > MAX_SCAN:
            break
        if human >= HUMAN_CAP and ai_total >= AI_TOTAL_CAP:
            break
        if not is_clean(row):
            continue
        model = row.get("model")
        text = (row.get("generation") or "").strip()
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
        rows_out.append([row.get("id"), model, label, row.get("domain"), text])
        if len(rows_out) % 100 == 0:
            print(f"scanned={scanned} kept={len(rows_out)} human={human} ai={ai_total}", flush=True)

    print(f"per_model={per_model}", flush=True)
    return rows_out


def write_csv(rows_out: list, path: str) -> None:
    """Write the subset using the frozen column contract."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "model", "label", "domain", "text"])
        writer.writerows(rows_out)


def label_counts(path: str) -> dict:
    """Count CSV rows grouped by the label column."""
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["label"]] += 1
    return dict(counts)


def main() -> None:
    """Build the RAID subset and print label counts."""
    rows_out = collect()
    write_csv(rows_out, OUT)
    print("\nDONE", flush=True)
    print(f"total rows: {len(rows_out)}", flush=True)
    print(f"label counts: {label_counts(OUT)}", flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
