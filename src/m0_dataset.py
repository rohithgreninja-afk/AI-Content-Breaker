"""M0: build a balanced clean MAGE subset at data/mage_clean.csv.

MAGE (yaful/MAGE, Li et al. 2024) pairs human text from 10 sources (Reddit
CMV, Yelp, XSum news, TLDR, ELI5, WritingPrompts, ROCStories, HellaSwag,
SQuAD, SciGen) with machine text from 27 LLMs. That breadth is the point: the
earlier RAID pull was 100% scientific abstracts from 6 older models, and every
human row was 96+ words, so the scorers learned "short or not an abstract =
AI".

The full train.csv is 400 MB and the HF CDN is slow from some networks, so
this samples through the HF datasets-server rows API instead: blocks of rows
at seeded random offsets across the whole split, cached to
data/mage_pool.jsonl, then balanced per (domain, label). The dataset revision
is pinned so a rebuild gives the same rows.

Frozen CSV contract: id, model, label, domain, text  (label in {human, ai}).
MAGE's own labels are 1 = human, 0 = machine; they are mapped here.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import Counter

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "data", "mage_clean.csv")
POOL = os.path.join(REPO_ROOT, "data", "mage_pool.jsonl")

DATASET = "yaful/MAGE"
# Pinned commit of yaful/MAGE; the rows API serves the current revision, so
# the pool records it and main() warns if the hub has moved on.
REVISION = "342663f0a2b775455c023f5d36a1341ff0ec5402"
SPLIT = "train"
ROWS_API = "https://datasets-server.huggingface.co/rows"
INFO_API = f"https://huggingface.co/api/datasets/{DATASET}"

N_BLOCKS = 200        # random offsets to read
BLOCK_LEN = 100       # rows per offset (the API maximum)
REQUEST_GAP = 1.0     # seconds between requests; the API rate-limits (429)
PER_CELL = 300        # rows kept per (domain, label), at most
MIN_WORDS = 20        # drop fragments; MIN_WORDS in pipeline gates at 50
MAX_WORDS = 300       # longer texts are cut to this many words
SEED = 42
LLAMA_SIZES = {"7B", "13B", "30B", "65B"}


def _fetch_block(offset: int) -> list[dict]:
    """One block of rows from the datasets-server API, with retries."""
    params = {"dataset": DATASET, "config": "default", "split": SPLIT,
              "offset": offset, "length": BLOCK_LEN}
    for attempt in range(8):
        try:
            response = requests.get(ROWS_API, params=params, timeout=60)
            if response.status_code == 200:
                return [
                    {"row_idx": item["row_idx"], **item["row"]}
                    for item in response.json().get("rows", [])
                ]
            status = response.status_code
        except requests.RequestException as error:
            status = type(error).__name__
        # 429 means slow down; back off hard rather than hammer the API.
        wait = min(15 * 2 ** attempt, 300)
        print(f"  offset {offset}: {status}, retrying in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"rows API kept failing at offset {offset}")


def _num_rows() -> int:
    """Rows in the split, from the rows API's own count."""
    params = {"dataset": DATASET, "config": "default", "split": SPLIT,
              "offset": 0, "length": 1}
    response = requests.get(ROWS_API, params=params, timeout=60)
    response.raise_for_status()
    return int(response.json()["num_rows_total"])


def fetch_pool() -> list[dict]:
    """Read N_BLOCKS blocks at seeded random offsets into POOL.

    POOL is appended block by block, so an interrupted run resumes where it
    stopped instead of re-reading everything.
    """
    rows: list[dict] = []
    if os.path.exists(POOL):
        with open(POOL, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
    have = {row["row_idx"] for row in rows}

    total = _num_rows()
    rng = random.Random(SEED)
    offsets = sorted(rng.sample(range(0, total - BLOCK_LEN), N_BLOCKS))
    todo = [offset for offset in offsets if offset not in have]
    if todo:
        print(f"fetching {len(todo)} of {N_BLOCKS} blocks of {BLOCK_LEN} "
              f"from {total} rows ...", flush=True)
    os.makedirs(os.path.dirname(POOL), exist_ok=True)
    with open(POOL, "a", encoding="utf-8") as handle:
        for i, offset in enumerate(todo, 1):
            for row in _fetch_block(offset):
                if row["row_idx"] not in have:  # blocks can overlap
                    have.add(row["row_idx"])
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} blocks, {len(rows)} rows", flush=True)
            time.sleep(REQUEST_GAP)
    return rows


def parse_src(src: str) -> tuple[str, str]:
    """Split a MAGE src tag into (domain, model).

    "cmv_human" -> ("cmv", "human");
    "squad_machine_continuation_flan_t5_xxl" -> ("squad", "flan_t5_xxl").
    Domains can contain "_" (sci_gen), so split on the human/machine marker.
    """
    if src.endswith("_human"):
        return src[: -len("_human")], "human"
    domain, _, rest = src.partition("_machine_")
    # rest is "<method>_<model>", method being continuation/topical/specified.
    method, _, model = rest.partition("_")
    model = model or method
    # MAGE tags LLaMA by size alone ("..._continuation_30B").
    if model in LLAMA_SIZES:
        model = f"llama_{model}"
    return domain, model


def is_clean(text: str) -> bool:
    """Keep real prose: at least MIN_WORDS words, mostly letters."""
    if len(text.split()) < MIN_WORDS:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) >= 0.6


def _cut(row: list, n_words: int) -> list:
    """Copy of a CSV row with its text cut to the first n_words words."""
    return row[:4] + [" ".join(row[4].split()[:n_words])]


def collect(pool: list[dict]) -> list[list]:
    """Balance and length-match the pool, domain by domain.

    Within each domain the classes get equal counts (so the domain says
    nothing about the label), and each human row is paired with an AI row
    and both are cut to the shorter of the two (so length says nothing about
    the label either). That was the RAID failure: every short text was AI.
    """
    cells: dict[tuple[str, str], list[list]] = {}
    for row in pool:
        text = " ".join(str(row["text"]).split())
        if not is_clean(text):
            continue
        domain, model = parse_src(row["src"])
        label = "human" if int(row["label"]) == 1 else "ai"
        cells.setdefault((domain, label), []).append(
            [f"mage-{SPLIT}-{row['row_idx']}", model, label, domain, text])

    rng = random.Random(SEED)
    rows_out: list[list] = []
    for domain in sorted({domain for domain, _ in cells}):
        human = cells.get((domain, "human"), [])
        ai = cells.get((domain, "ai"), [])
        rng.shuffle(human)
        rng.shuffle(ai)
        n_pairs = min(len(human), len(ai), PER_CELL)
        for human_row, ai_row in zip(human[:n_pairs], ai[:n_pairs]):
            n_words = min(len(human_row[4].split()), len(ai_row[4].split()),
                          MAX_WORDS)
            rows_out.append(_cut(human_row, n_words))
            rows_out.append(_cut(ai_row, n_words))
        print(f"  {domain:10} human={len(human):5} ai={len(ai):5} "
              f"pairs kept={n_pairs}", flush=True)
    rng.shuffle(rows_out)
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
    """Build the MAGE subset and print label counts."""
    try:
        live = requests.get(INFO_API, timeout=30).json().get("sha")
        if live and live != REVISION:
            print(f"warning: {DATASET} is now at {live}, pinned {REVISION}; "
                  "the rows API serves the live revision", flush=True)
    except requests.RequestException:
        pass
    pool = fetch_pool()
    print(f"pool: {len(pool)} rows", flush=True)
    rows_out = collect(pool)
    write_csv(rows_out, OUT)
    print(f"\ntotal rows: {len(rows_out)}", flush=True)
    print(f"label counts: {label_counts(OUT)}", flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
