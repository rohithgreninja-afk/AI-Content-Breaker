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
rows at seeded random offsets across the available API split, cached and
resumable in data/<source>_pool.jsonl. The API can expose a partial split
(currently RAID); the manifest records this and the observed domains.
Revisions are checked on the hub and on every rows-API response.

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
        if isinstance(answers, str):
            return answers or None
    if not isinstance(answers, (list, tuple)):
        return None
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


def _read_pool(path: str) -> list[dict]:
    """Recover only an interrupted final JSONL write, rejecting other damage."""
    rows = []
    truncate_at = None
    needs_newline = False
    with open(path, "rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n") or handle.read(1):
                    raise RuntimeError(f"corrupt cached JSONL record at byte {start}")
                truncate_at = start
                break
            needs_newline = not line.endswith(b"\n")
    if truncate_at is not None:
        with open(path, "r+b") as handle:
            handle.truncate(truncate_at)
    elif needs_newline:
        with open(path, "ab") as handle:
            handle.write(b"\n")
    return rows


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
        rows = _read_pool(path)
    have = {row["row_idx"] for row in rows}
    if len(have) != len(rows):
        raise RuntimeError(f"{name}: duplicate row indices in cached pool")

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
        if metadata.get("sample_offsets", offsets) != offsets:
            raise RuntimeError(f"{name}: requested sample differs from cached sample")
        if metadata.get("pool_sha256") and metadata["pool_sha256"] != file_sha256(path):
            raise RuntimeError(f"{name}: cached pool digest does not match metadata")
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
        if rows and len(completed) != len(offsets):
            raise RuntimeError("legacy MAGE cache is not the complete seeded sample")
        if have - expected:
            # An abandoned fetch with a different seed may have appended extra
            # rows. Keep exactly the original deterministic complete sample.
            rows = [row for row in rows if row["row_idx"] in expected]
            have = expected
            fd, temporary = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
    metadata = {**identity, "sample_offsets": offsets,
                "completed_offsets": sorted(completed)}
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


def _text_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def natural_group(name: str, row: dict, text: str) -> str:
    """Keep multiple outputs from one original document in the same split."""
    if name == "hc3":
        key = str(row.get("id") or _text_key(str(row.get("question", ""))))
    elif name == "wiki":
        key = str(row.get("id") or row.get("title") or row["row_idx"])
    elif name == "raid":
        key = str(row.get("source_id") or row.get("id") or row["row_idx"])
    else:
        # MAGE does not expose the original document ID. Continuations often
        # share their opening with the human text or other generators.
        key = _text_key(" ".join(text.split()[:20]))
    return f"{name}:{key}"


def collect(name: str, pool: list[dict],
            row_groups: dict[str, str] | None = None,
            group_links: list[tuple[str, str]] | None = None) -> list[list]:
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
            if label not in ("human", "ai"):
                raise ValueError(f"invalid {name} label: {label!r}")
            if (not isinstance(text, str) or not isinstance(domain, str)
                    or not domain.strip() or not model):
                continue
            text = " ".join(text.split())
            row_id = f"{name}-{row['row_idx']}-{index}"
            group = natural_group(name, row, text)
            if row_groups is not None:
                row_groups[row_id] = group
            # Link every raw document before balancing or deduplication. This
            # catches shared prompts across corpora (e.g. MAGE/HC3 ELI5), and
            # preserves chains through records that are not finally selected.
            if group_links is not None and len(text.split()) >= 20:
                prefix = _text_key(" ".join(text.split()[:20]))
                group_links.append((group, "prefix:" + prefix))
            if not is_clean(text):
                continue
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
    """Atomically write the subset using the frozen column contract."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "model", "label", "domain", "text"])
            writer.writerows(rows_out)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def deduplicate_pairs(rows: list[list], natural_groups: dict[str, str],
                      group_links: list[tuple[str, str]] | None = None
                      ) -> tuple[list[list], dict[str, str], int]:
    """Deduplicate after truncation, dropping complete matched pairs.

    Union source groups through each length-matched pair so related outputs
    and their matching partners can never straddle evaluation splits.
    """
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    for left, right in group_links or []:
        union(left, right)
    seen: dict[str, str] = {}
    retained: list[list] = []
    dropped = 0
    if len(rows) % 2:
        raise ValueError("balanced rows must contain complete human/AI pairs")
    for i in range(0, len(rows), 2):
        human, ai = rows[i:i + 2]
        if (human[2] != "human" or ai[2] != "ai" or human[3] != ai[3]
                or len(human[4].split()) != len(ai[4].split())):
            raise ValueError("invalid length-matched pair")
        left, right = _text_key(human[4]), _text_key(ai[4])
        human_group, ai_group = natural_groups[human[0]], natural_groups[ai[0]]
        union(human_group, ai_group)
        for key, group in ((left, human_group), (right, ai_group)):
            if key in seen:
                union(group, seen[key])
        if left == right or left in seen or right in seen:
            dropped += 1
            continue
        seen.update({left: human_group, right: ai_group})
        retained.extend((human, ai))
    return retained, {row[0]: find(natural_groups[row[0]]) for row in retained}, dropped


def write_manifest(rows: list[list], row_groups: dict[str, str],
                   names: list[str], dropped_pairs: int) -> dict:
    """Record source provenance and the exact CSV-to-evaluation-group map."""
    sources = {}
    for name in names:
        dataset, config, split, revision, n_blocks, _ = SOURCES[name]
        pool_path = os.path.join(DATA_DIR, f"{name}_pool.jsonl")
        metadata_path = os.path.join(DATA_DIR, f"{name}_pool.meta.json")
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        selected = [row for row in rows if row[3].startswith(name + ":")]
        sources[name] = {
            "dataset": dataset, "url": f"https://huggingface.co/datasets/{dataset}",
            "config": config, "split": split, "revision": revision,
            "blocks": n_blocks, "pool_rows": metadata["pool_rows"],
            "pool_sha256": file_sha256(pool_path),
            "api_num_rows_total": metadata["num_rows_total"],
            "api_partial": metadata["partial"], "rows": len(selected),
            "labels": dict(Counter(row[2] for row in selected)),
            "domains": dict(sorted(Counter(row[3] for row in selected).items())),
        }
    manifest = {
        "schema_version": 1, "csv_path": "data/train_clean.csv",
        "csv_sha256": file_sha256(OUT), "rows": len(rows),
        "seed": SEED, "min_words": MIN_WORDS, "max_words": MAX_WORDS,
        "max_pairs_per_domain": PER_CELL,
        "duplicate_pairs_dropped": dropped_pairs,
        "labels": dict(Counter(row[2] for row in rows)), "sources": sources,
        "grouping": "natural source IDs, length-matched pairs, and globally shared normalized first 20 words from all raw documents",
        "row_groups": row_groups,
    }
    _atomic_json(manifest, os.path.join(DATA_DIR, "train_manifest.json"))
    return manifest


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
    natural_groups: dict[str, str] = {}
    group_links: list[tuple[str, str]] = []
    for name in names:
        check_revision(name)
        pool = fetch_pool(name)
        print(f"{name}: pool {len(pool)} rows", flush=True)
        rows_out.extend(collect(name, pool, natural_groups, group_links))
    rows_out, row_groups, dropped_pairs = deduplicate_pairs(
        rows_out, natural_groups, group_links)
    random.Random(SEED).shuffle(rows_out)
    write_csv(rows_out, OUT)
    write_manifest(rows_out, row_groups, names, dropped_pairs)
    by_source = Counter(row[3].split(":")[0] for row in rows_out)
    print(f"\ntotal rows: {len(rows_out)} {dict(by_source)}", flush=True)
    print(f"label counts: {label_counts(OUT)}", flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
