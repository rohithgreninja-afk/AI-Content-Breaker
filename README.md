# AI Content Breaker

Compiler-design course project: an AI-generated-text detector that maps the
compiler front-end (lexing, symbol table, CFG parsing, automata, attribute
grammars) onto a text-analysis pipeline.

**Author:** Rohith Ram V, VIT, Compiler Design (BCSE307P/L).

This repo is built module-by-module (M0..M7). See `PROJECT_CONTEXT.md` for
architecture and interface contracts, and `PROGRESS.md` for what currently
exists and how to run it.

## Setup

Python 3.14 in `.venv` at the repo root. Install deps with the venv interpreter:

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Current status

M0 (dataset pull) and M1 (lexer) are in place. Later modules are not started yet.

## Layout

- `src/` — pipeline modules (`m0_dataset.py`, `m1_lexer.py`, …)
- `tests/` — smoke tests per module
- `scripts/` — exploratory RAID probes (`pull_raid.py`, etc.)
- `data/` — generated `raid_clean.csv` (not redistributed)
- `models/` — trained classifier artifacts (later)
