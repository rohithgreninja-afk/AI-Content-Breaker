"""Smoke tests for the M1 maximal-munch lexer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m1_lexer import tokenize  # noqa: E402


class TestM1Lexer(unittest.TestCase):
    """Plain-text cases covering the four token types."""

    def test_simple_sentence(self) -> None:
        tokens = tokenize("Hello, world!")
        self.assertEqual(
            [(t["type"], t["value"], t["idx"]) for t in tokens],
            [
                ("WORD", "Hello", 0),
                ("PUNCT", ",", 1),
                ("WORD", "world", 2),
                ("SENTENCE_END", "!", 3),
            ],
        )

    def test_number_maximal_munch(self) -> None:
        """3.14 is one NUMBER, not NUMBER + SENTENCE_END + NUMBER."""
        tokens = tokenize("pi is 3.14.")
        self.assertEqual(
            [(t["type"], t["value"]) for t in tokens],
            [
                ("WORD", "pi"),
                ("WORD", "is"),
                ("NUMBER", "3.14"),
                ("SENTENCE_END", "."),
            ],
        )

    def test_integer_then_sentence_end(self) -> None:
        """42. with no fractional digits is NUMBER then SENTENCE_END."""
        tokens = tokenize("42.")
        self.assertEqual(
            [(t["type"], t["value"]) for t in tokens],
            [("NUMBER", "42"), ("SENTENCE_END", ".")],
        )

    def test_whitespace_skipped(self) -> None:
        tokens = tokenize("  one   two\nthree\t")
        self.assertEqual([t["value"] for t in tokens], ["one", "two", "three"])
        self.assertTrue(all(t["type"] == "WORD" for t in tokens))

    def test_idx_is_stream_position(self) -> None:
        tokens = tokenize("A 1.")
        self.assertEqual([t["idx"] for t in tokens], [0, 1, 2])

    def test_empty_and_whitespace_only(self) -> None:
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(" \n\t "), [])


if __name__ == "__main__":
    unittest.main()
