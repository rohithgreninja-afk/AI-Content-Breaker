"""M0: build a balanced, length-matched training set at data/train_clean.csv.

Four public human-vs-machine datasets, each sampled and balanced the same way:

    mage  yaful/MAGE                  10 domains (Reddit CMV, Yelp, XSum, TLDR,
                                      ELI5, WritingPrompts, ROCStories,
                                      HellaSwag, SQuAD, SciGen), 27 LLMs
    hc3   Hello-SimpleAI/HC3          human vs ChatGPT answers to the same
                                      questions (ELI5, open QA, CS wiki,
                                      medicine, finance)
    wiki  aadityaubhat/GPT-wiki-intro Wikipedia intros vs GPT-3 (curie)
                                      intros for the same article
    raid  liamdugan/raid              8 domains (abstracts, books, news,
                                      poetry, recipes, reddit, reviews, wiki),
                                      11 generators incl. GPT-4 and ChatGPT;
                                      unattacked rows only

Breadth is the point. The first RAID pull read only the start of a sorted
file, so it was 100% scientific abstracts with every human row 96+ words,
and the scorers learned "short or not an abstract = AI".

Full downloads are hundreds of MB and the HF CDN is slow from some networks,
so every source is sampled through the HF datasets-server rows API: blocks of
rows at seeded random offsets across the whole split, cached and resumable in
data/<source>_pool.jsonl. Revisions are pinned below and checked at start.

Balancing, per (source, domain): equal human and AI counts (at most PER_CELL
pairs), and each human text is paired with an AI text and both are cut to the
shorter length, so neither domain nor length can stand in for the label.

Frozen CSV contract: id, model, label, domain, text  (label in {human, ai}).
The domain column is "<source>:<domain>", e.g. "hc3:finance".
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(DATA_DIR, "train_clean.csv")

ROWS_API = "https://datasets-server.huggingface.co/rows"
INFO_API = "https://huggingface.co/api/datasets/"

BLOCK_LEN = 100       # rows per offset (the API maximum)
REQUEST_GAP = 1.0     # seconds between requests; the API rate-limits (429)
PER_CELL = 300        # human/AI pairs kept per (source, domain), at most
MIN_WORDS = 20        # drop fragments; MIN_WORDS in pipeline gates at 50
MAX_WORDS = 300       # longer texts are cut to this many words
SEED = 42
LLAMA_SIZES = {"7B", "13B", "30B", "65B"}


# --------------------------------------------------------------------------
# Per-source row mappers: one API row -> list of (label, model, domain, text)
# --------------------------------------------------------------------------

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


def map_mage(row: dict) -> list[tuple]:
    """MAGE labels are 1 = human, 0 = machine."""
    if row.get("label") not in (0, 1, "0", "1"):
        raise ValueError(f"invalid MAGE label: {row.get('label')!r}")
    domain, model = parse_src(row["src"])
    label = "human" if int(row["label"]) == 1 else "ai"
    return [(label, model, domain, row["text"])]


def _first(answers) -> str | None:
    """First answer of an HC3 answer list (sometimes served as a string)."""
    if isinstance(answers, str):
        try:
            answers = ast.literal_eval(answers)
        except (ValueError, SyntaxError):
            return answers or None
    return next((answer for answer in (answers or [])
                 if isinstance(answer, str) and answer.strip()), None)


def map_hc3(row: dict) -> list[tuple]:
    """One question gives a human answer and a ChatGPT answer."""
    out = []
    human = _first(row.get("human_answers"))
    machine = _first(row.get("chatgpt_answers"))
    if human:
        out.append(("human", "human", row["source"], human))
    if machine:
        out.append(("ai", "chatgpt", row["source"], machine))
    return out


def map_wiki(row: dict) -> list[tuple]:
    """One article gives the real intro and a GPT-3 curie intro."""
    return [("human", "human", "wikipedia", row["wiki_intro"]),
            ("ai", "gpt3_curie", "wikipedia", row["generated_intro"])]


def map_raid(row: dict) -> list[tuple]:
    """Unattacked RAID rows only; model "human" marks human text."""
    if (row.get("attack") != "none" or not row.get("generation")
            or not row.get("model") or not row.get("domain")):
        return []
    label = "human" if row["model"] == "human" else "ai"
    return [(label, row["model"], row["domain"], row["generation"])]


# name -> dataset, config, split, pinned revision, blocks to read, mapper
SOURCES = {
    "mage": ("yaful/MAGE", "default", "train",
             "342663f0a2b775455c023f5d36a1341ff0ec5402", 200, map_mage),
    "hc3": ("Hello-SimpleAI/HC3", "all", "train",
            "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3", 40, map_hc3),
    "wiki": ("aadityaubhat/GPT-wiki-intro", "default", "train",
             "017258d41991489752363bfa4cbcc12725786967", 20, map_wiki),
    # RAID is 2.27M rows, about 1 in 12 unattacked, so it needs more blocks.
    "raid": ("liamdugan/raid", "raid", "train",
             "865cac74188466cb0c3b7574a10204007b57a459", 300, map_raid),
}


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _get_rows(dataset: str, config: str, split: str, offset: int,
              length: int, revision: str | None = None) -> dict:
    """One rows-API call, with backoff on rate limits and server errors."""
    params = {"dataset": dataset, "config": config, "split": split,
              "offset": offset, "length": length}
    for attempt in range(8):
        try:
            response = requests.get(ROWS_API, params=params, timeout=90)
            if response.status_code == 200:
                if revision and response.headers.get("x-revision") != revision:
                    raise RuntimeError(
                        f"{dataset}: rows API revision "
                        f"{response.headers.get('x-revision')!r} does not match "
                        f"pinned {revision}; refusing to mix dataset versions")
                return response.json()
            status = response.status_code
        except requests.RequestException as error:
            status = type(error).__name__
        # 429 means slow down; back off hard rather than hammer the API.
        wait = min(15 * 2 ** attempt, 300)
        print(f"  {dataset} offset {offset}: {status}, retrying in {wait}s",
              flush=True)
        time.sleep(wait)
    raise RuntimeError(f"rows API kept failing: {dataset} offset {offset}")


def check_revision(name: str) -> None:
    """Fail closed if the live rows API cannot serve the pinned dataset."""
    dataset, _, _, revision, _, _ = SOURCES[name]
    try:
        response = requests.get(INFO_API + dataset, timeout=30)
        response.raise_for_status()
        live = response.json().get("sha")
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f"cannot verify {dataset} revision") from error
    if live != revision:
        raise RuntimeError(f"{dataset} is at {live}, expected pinned {revision}")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def fetch_pool(name: str) -> list[dict]:
    """Read one source's blocks at seeded random offsets into its pool file.

    The pool is appended block by block, so an interrupted run resumes where
    it stopped instead of re-reading everything.
    """
    dataset, config, split, revision, n_blocks, _ = SOURCES[name]
    path = os.path.join(DATA_DIR, f"{name}_pool.jsonl")
    metadata_path = os.path.join(DATA_DIR, f"{name}_pool.meta.json")
    rows: list[dict] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
    have = {row["row_idx"] for row in rows}

    info = _get_rows(dataset, config, split, 0, 1, revision)
    total = int(info["num_rows_total"])
    # Same seed for every source: offsets differ anyway because the sizes do,
    # and it keeps the original MAGE pool's offsets valid.
    rng = random.Random(SEED)
    offsets = sorted(rng.sample(range(0, total - BLOCK_LEN), n_blocks))
    identity = {"dataset": dataset, "config": config, "split": split,
                "revision": revision, "num_rows_total": total,
                "partial": bool(info.get("partial", False)),
                "seed": SEED, "block_length": BLOCK_LEN}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"{name}: cache identity differs from pinned source")
        completed = set(metadata.get("completed_offsets", []))
        for offset in completed:
            if not all(i in have for i in range(offset, offset + BLOCK_LEN)):
                raise RuntimeError(f"{name}: cache is missing rows from block {offset}")
    else:
        # The existing MAGE cache predates metadata. Adopt it only when its
        # complete seeded sample is present and both hub/API revisions match.
        if rows and name != "mage":
            raise RuntimeError(f"{name}: nonempty legacy cache lacks provenance")
        completed = {offset for offset in offsets
                     if all(i in have for i in range(offset, offset + BLOCK_LEN))}
        expected = {i for offset in offsets for i in range(offset, offset + BLOCK_LEN)}
        if rows and (len(completed) != len(offsets) or have != expected):
            raise RuntimeError("legacy MAGE cache is not the complete seeded sample")
    metadata = {**identity, "completed_offsets": sorted(completed)}
    _atomic_json(metadata, metadata_path)
    todo = [offset for offset in offsets if offset not in completed]
    if todo:
        print(f"{name}: fetching {len(todo)} of {n_blocks} blocks from "
              f"{total} rows ...", flush=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for i, offset in enumerate(todo, 1):
            page = _get_rows(dataset, config, split, offset, BLOCK_LEN, revision)
            page_rows = page.get("rows", [])
            if {item["row_idx"] for item in page_rows} != set(range(offset, offset + BLOCK_LEN)):
                raise RuntimeError(f"{name}: incomplete API block at {offset}")
            for item in page_rows:
                if item["row_idx"] not in have:  # blocks can overlap
                    have.add(item["row_idx"])
                    row = {"row_idx": item["row_idx"], **item["row"]}
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed.add(offset)
            metadata["completed_offsets"] = sorted(completed)
            _atomic_json(metadata, metadata_path)
            if i % 10 == 0 or i == len(todo):
                print(f"  {name}: {i}/{len(todo)} blocks, {len(rows)} rows",
                      flush=True)
            time.sleep(REQUEST_GAP)
    metadata["pool_sha256"] = file_sha256(path)
    metadata["pool_rows"] = len(rows)
    _atomic_json(metadata, metadata_path)
    return rows


# --------------------------------------------------------------------------
# Cleaning and balancing
# --------------------------------------------------------------------------

def is_clean(text: str) -> bool:
    """Keep real prose: at least MIN_WORDS words, mostly letters."""
    if len(text.split()) < MIN_WORDS:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) >= 0.6


def _cut(row: list, n_words: int) -> list:
    """Copy of a CSV row with its text cut to the first n_words words."""
    return row[:4] + [" ".join(row[4].split()[:n_words])]


def collect(name: str, pool: list[dict]) -> list[list]:
    """Balance and length-match one source's pool, domain by domain.

    Within each domain the classes get equal counts (so the domain says
    nothing about the label), and each human row is paired with an AI row
    and both are cut to the shorter of the two (so length says nothing about
    the label either).
    """
    mapper = SOURCES[name][5]
    cells: dict[tuple[str, str], list[list]] = {}
    for row in pool:
        for index, (label, model, domain, text) in enumerate(mapper(row)):
            text = " ".join(str(text).split())
            if not is_clean(text):
                continue
            row_id = f"{name}-{row['row_idx']}-{index}"
            cells.setdefault((domain, label), []).append(
                [row_id, model, label, f"{name}:{domain}", text])

    rng = random.Random(f"{SEED}-{name}")
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
        print(f"  {name}:{domain:14} human={len(human):5} ai={len(ai):5} "
              f"pairs kept={n_pairs}", flush=True)
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


def main(argv: list[str] | None = None) -> None:
    """Build the combined set. `--only mage,hc3` limits the sources."""
    argv = list(sys.argv[1:] if argv is None else argv)
    names = list(SOURCES)
    if "--only" in argv:
        names = argv[argv.index("--only") + 1].split(",")
    rows_out: list[list] = []
    for name in names:
        check_revision(name)
        pool = fetch_pool(name)
        print(f"{name}: pool {len(pool)} rows", flush=True)
        rows_out.extend(collect(name, pool))
    random.Random(SEED).shuffle(rows_out)
    write_csv(rows_out, OUT)
    by_source = Counter(row[3].split(":")[0] for row in rows_out)
    print(f"\ntotal rows: {len(rows_out)} {dict(by_source)}", flush=True)
    print(f"label counts: {label_counts(OUT)}", flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
