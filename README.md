# AI Content Breaker

Compiler-design course project: an AI-generated-text detector that maps the
compiler front-end (lexing, symbol table, CFG parsing, automata, attribute
grammars) onto a text-analysis pipeline.



This repo is built module-by-module (M0..M7). See `PROJECT_CONTEXT.md` for
architecture and interface contracts, and `PROGRESS.md` for what currently
exists and how to run it.

## Setup

Python 3.14 in `.venv` at the repo root. Install deps with the venv interpreter:

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Current status

M0–M5 are in place, with 112 tests passing. On the 760-row RAID sample
(400 human / 360 AI):

| scorer | accuracy | notes |
|---|---|---|
| M5a rule-based | 0.816 held-out | 18 thresholded rules, fully inspectable |
| M5b logistic regression | 0.847 test | ROC-AUC 0.941, 5-fold CV 0.867 ± 0.031 |

## Pipeline

| module | does | public entry point |
|---|---|---|
| `m0_dataset.py` | streams + balances RAID into CSV | `main()` |
| `m1_lexer.py` | maximal-munch lexer | `tokenize(text)` |
| `m2_symtab.py` | symbol table over the token stream | `build_symtab(tokens)` |
| `m3_parser.py` | recursive-descent CFG **and** template DFA | `parse_text(text)` |
| `m4_features.py` | 89-feature vector from M1–M3 | `extract_features(text)` |
| `m5_scorer.py` | rule scorer + trained classifier | `RuleScorer`, `ModelScorer` |
| `pipeline.py` | runs M1 to M5 on one text, stage by stage | `run_pipeline(text)` |

## Running it

```
.venv\Scripts\python.exe src\m0_dataset.py          # regenerate data/raid_clean.csv
.venv\Scripts\python.exe src\m5_scorer.py --train   # fit, report, save the model
.venv\Scripts\python.exe src\m5_scorer.py --score path\to\text.txt
.venv\Scripts\python.exe src\m5_scorer.py --audit   # per-rule lift, for tuning
.venv\Scripts\python.exe src\pipeline.py --test 1     # one held-out test row through M1-M5, stage by stage
.venv\Scripts\python.exe src\pipeline.py --eval       # all held-out test rows: accuracy for both scorers
.venv\Scripts\python.exe -m unittest discover -s tests  # run all 112 tests
```

Each module also runs standalone as a demo on one real row, e.g.
`.venv\Scripts\python.exe src\m3_parser.py`.

First run downloads the nltk POS tagger (`averaged_perceptron_tagger_eng`).
If that is unavailable, M3 falls back to a built-in rule tagger so the
pipeline still runs offline.

## Layout

- `src/` — pipeline modules (`m0_dataset.py` … `m5_scorer.py`)
- `tests/` — smoke tests per module (run any file directly)
- `scripts/` — exploratory RAID probes (`pull_raid.py`, etc.)
- `data/` — generated `raid_clean.csv` (gitignored, not redistributed)
- `models/` — trained classifier artifacts (gitignored, not committed; regenerate with `.venv\Scripts\python.exe src\m5_scorer.py --train`)
