"""M2: symbol table over an M1 token stream.

Maps each word identity to {count, positions}, where positions are the token
idx values from the lexer (stream positions, not compacted word-only indexes).

Decisions (frozen for later modules):
- Identity is case-folded: WORD tokens are keyed by value.lower().
- Only WORD tokens are indexed. NUMBER, PUNCT, and SENTENCE_END are skipped
  so lexical_diversity / transition_phrase_density stay word-based.
"""

from __future__ import annotations

# WORD only: numbers are literals, not identifier-like word identities.
INDEXED_TYPES = {"WORD"}


def word_key(token: dict) -> str:
    """Case-folded identity for a WORD token."""
    return token["value"].lower()


def build_symtab(tokens: list[dict]) -> dict[str, dict]:
    """Build the symbol table from an M1 token stream.

    Returns dict word -> {count: int, positions: list[int]}.
    """
    sym_tab: dict[str, dict] = {}
    for token in tokens:
        if token["type"] not in INDEXED_TYPES:
            continue
        key = word_key(token)
        entry = sym_tab.get(key)
        if entry is None:
            entry = {"count": 0, "positions": []}
            sym_tab[key] = entry
        entry["count"] += 1
        entry["positions"].append(token["idx"])
    return sym_tab


def top_words(sym_tab: dict[str, dict], n: int = 10) -> list[tuple[str, int]]:
    """Return up to n (word, count) pairs, highest count first, then A-Z."""
    ranked = sorted(
        ((word, entry["count"]) for word, entry in sym_tab.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:n]


def main() -> None:
    """Pipe one MAGE row through M1 then M2 and print the top 10 words."""
    import csv
    import os

    from m1_lexer import tokenize

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "data", "mage_clean.csv")
    with open(csv_path, encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    tokens = tokenize(row["text"])
    sym_tab = build_symtab(tokens)
    print(f"id={row['id']} label={row['label']} domain={row['domain']}")
    print(f"tokens={len(tokens)} unique_words={len(sym_tab)}")
    print("top 10 words by count:")
    for word, count in top_words(sym_tab, 10):
        positions = sym_tab[word]["positions"]
        print(f"  {word!r:20} count={count:3} positions={positions[:8]}")


if __name__ == "__main__":
    main()
