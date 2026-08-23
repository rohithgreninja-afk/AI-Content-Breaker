# PROGRESS

## M0 - Dataset (DONE, verified)
- `src/m0_dataset.py` builds `data/raid_clean.csv` from the RAID benchmark.
- Switched from the datasets-server rows API (kept 429-rate-limiting) to
  **streaming** `liamdugan/raid` via the `datasets` library with a shuffled
  buffer. Robust and finishes in ~2 min.
- Clean rows only (`attack == "none"`, text length >= 40). MVP caps: 400 human,
  400 AI, 60 per model. Frozen columns: id, model, label, domain, text.
- Result: **760 rows (400 human / 360 AI)** across 6 models (llama-chat, gpt2,
  mpt, mistral, mpt-chat, mistral-chat). CSV is gitignored (regenerable).
- Run: `.venv\Scripts\python.exe src\m0_dataset.py`
- To scale up later: raise HUMAN_CAP / AI_TOTAL_CAP / AI_PER_MODEL_CAP.

## M1 - Lexer (DONE, verified)
- `src/m1_lexer.py`: maximal-munch scanner. `tokenize(text) -> list[{type,value,idx}]`.
- Types: WORD, NUMBER, PUNCT, SENTENCE_END. Whitespace skipped. `idx` is the
  0-based stream position.
- Verified on a real RAID row: 299 tokens, correct WORD/PUNCT/SENTENCE_END split.
- Run demo: `.venv\Scripts\python.exe src\m1_lexer.py`
- Tests: `tests/test_m1_lexer.py`

## Environment notes
- Python 3.14.7 in `.venv`. Use `.venv\Scripts\python.exe`.
- Packages installed clean on 3.14: requests, nltk, scikit-learn, numpy, datasets.
- HF rows API is rate-limited; prefer streaming for any future dataset work.

## M2 - Symbol table (DONE, verified)
- `src/m2_symtab.py`: `build_symtab(tokens) -> dict word -> {count, positions}`.
- **Identity:** WORD values are case-folded with `.lower()` so `The`/`the` merge.
- **Indexed types:** WORD only. NUMBER, PUNCT, and SENTENCE_END are not entered
  (lexical_diversity and transition_phrase_density stay word-based).
- **Positions:** M1 token `idx` values (stream positions; gaps from skipped types).
- Tests: 5 cases in `tests/test_m2_symtab.py` (all passed).
- Real RAID row (`id=7db2f15a-...`, ai/abstracts): 299 tokens, 157 unique words.
  Top counts: the 19, of 14, and 8, to 7, a 6, our 6, graph 5, images 5, we 5,
  feature 4. Positions for `the` start at lexer idxs [51, 61, 109, ...].
- Run demo: `.venv\Scripts\python.exe src\m2_symtab.py`
- Run tests: `.venv\Scripts\python.exe tests\test_m2_symtab.py`

## Next: M3 - Dual syntax
- (a) recursive-descent CFG parser -> per-sentence parse trees
- (b) DFA over POS-tag sequences -> template signal
