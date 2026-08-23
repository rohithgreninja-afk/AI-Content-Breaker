"""Smoke tests for the M2 symbol table."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m1_lexer import tokenize  # noqa: E402
from m2_symtab import build_symtab, top_words  # noqa: E402


class TestM2Symtab(unittest.TestCase):
    """Identity, counts, and idx positions over a token stream."""

    def test_case_folding_merges_identity(self) -> None:
        tokens = tokenize("The cat sat. the CAT ran.")
        sym_tab = build_symtab(tokens)
        self.assertIn("the", sym_tab)
        self.assertNotIn("The", sym_tab)
        self.assertNotIn("CAT", sym_tab)
        self.assertEqual(sym_tab["the"]["count"], 2)
        self.assertEqual(sym_tab["cat"]["count"], 2)

    def test_only_word_tokens_are_indexed(self) -> None:
        tokens = tokenize("See 42 cats, 42!")
        # M1: WORD See, NUMBER 42, WORD cats, PUNCT ,, NUMBER 42, SENTENCE_END !
        self.assertEqual(
            [(t["type"], t["value"], t["idx"]) for t in tokens],
            [
                ("WORD", "See", 0),
                ("NUMBER", "42", 1),
                ("WORD", "cats", 2),
                ("PUNCT", ",", 3),
                ("NUMBER", "42", 4),
                ("SENTENCE_END", "!", 5),
            ],
        )
        sym_tab = build_symtab(tokens)
        self.assertEqual(set(sym_tab), {"see", "cats"})
        self.assertNotIn("42", sym_tab)
        self.assertNotIn(",", sym_tab)
        self.assertNotIn("!", sym_tab)

    def test_positions_are_token_idx(self) -> None:
        """Positions keep lexer idx, including gaps left by skipped types."""
        tokens = tokenize("A, a.")
        # WORD A idx=0, PUNCT , idx=1, WORD a idx=2, SENTENCE_END . idx=3
        sym_tab = build_symtab(tokens)
        self.assertEqual(sym_tab["a"]["count"], 2)
        self.assertEqual(sym_tab["a"]["positions"], [0, 2])

    def test_empty_stream(self) -> None:
        self.assertEqual(build_symtab([]), {})
        self.assertEqual(build_symtab(tokenize("   ")), {})

    def test_top_words_order(self) -> None:
        tokens = tokenize("to be or not to be")
        ranked = top_words(build_symtab(tokens), 3)
        self.assertEqual(ranked, [("be", 2), ("to", 2), ("not", 1)])


if __name__ == "__main__":
    unittest.main()
