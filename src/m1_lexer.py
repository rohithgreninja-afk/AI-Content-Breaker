"""M1: maximal-munch lexer. Raw text -> token stream {type, value, idx}.

Token types: WORD, NUMBER, PUNCT, SENTENCE_END.
Whitespace is skipped and never emitted.
"""

from __future__ import annotations

import string

# Sentence terminators are a dedicated type; other punctuation is PUNCT.
SENTENCE_END_CHARS = set(".!?")
# Characters that are clearly punctuation but not sentence ends.
EXTRA_PUNCT = set(string.punctuation) - SENTENCE_END_CHARS


def match_word(text: str, start: int) -> str | None:
    """Longest run of alphabetic characters starting at start, or None."""
    index = start
    while index < len(text) and text[index].isalpha():
        index += 1
    if index == start:
        return None
    return text[start:index]


def match_number(text: str, start: int) -> str | None:
    """Longest number at start: digits, optionally '.' plus more digits."""
    index = start
    while index < len(text) and text[index].isdigit():
        index += 1
    if index == start:
        return None
    # Maximal munch: absorb a fractional part when it actually has digits.
    if index < len(text) and text[index] == ".":
        frac = index + 1
        while frac < len(text) and text[frac].isdigit():
            frac += 1
        if frac > index + 1:
            index = frac
    return text[start:index]


def match_sentence_end(text: str, start: int) -> str | None:
    """One sentence-ending mark (. ! ?) at start, or None."""
    if start < len(text) and text[start] in SENTENCE_END_CHARS:
        return text[start]
    return None


def match_punct(text: str, start: int) -> str | None:
    """One non-sentence punctuation character at start, or None."""
    if start >= len(text):
        return None
    char = text[start]
    if char in EXTRA_PUNCT:
        return char
    # Leftover non-space, non-alnum symbols (e.g. quotes, currency, emoji).
    if not char.isspace() and not char.isalnum():
        return char
    return None


def longest_match(text: str, start: int) -> tuple[str, str] | None:
    """Apply maximal munch: pick the longest candidate; ties prefer WORD, NUMBER, SENTENCE_END, PUNCT."""
    candidates: list[tuple[str, str]] = []
    for token_type, matcher in (
        ("WORD", match_word),
        ("NUMBER", match_number),
        ("SENTENCE_END", match_sentence_end),
        ("PUNCT", match_punct),
    ):
        value = matcher(text, start)
        if value is not None:
            candidates.append((token_type, value))
    if not candidates:
        return None
    # Length first (maximal munch), then the order above as a stable tie-break.
    type_rank = {"WORD": 0, "NUMBER": 1, "SENTENCE_END": 2, "PUNCT": 3}
    candidates.sort(key=lambda item: (-len(item[1]), type_rank[item[0]]))
    return candidates[0]


def tokenize(text: str) -> list[dict]:
    """Scan text left to right and return the Token stream.

    Each token is {type, value, idx} with idx as the 0-based stream position.
    """
    tokens: list[dict] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        picked = longest_match(text, index)
        if picked is None:
            # Should not happen; skip one char so the scanner cannot hang.
            index += 1
            continue
        token_type, value = picked
        tokens.append({"type": token_type, "value": value, "idx": len(tokens)})
        index += len(value)
    return tokens


def main() -> None:
    """Tiny demo so the module can be run by itself."""
    sample = "Hello, world! 42 cats."
    tokens = tokenize(sample)
    print(f"input: {sample!r}")
    for token in tokens:
        print(token)


if __name__ == "__main__":
    main()
