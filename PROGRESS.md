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

## Next: M2 - Symbol table
- Build word identity/frequency/positions over the M1 token stream.
- Contract: dict word -> {count: int, positions: list[int]}.
