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

M0–M5 are in place, with 112 tests passing. Training data is a balanced,
length-matched sample of [MAGE](https://huggingface.co/datasets/yaful/MAGE):
4734 rows (2367 human / 2367 AI), 10 domains (Reddit CMV, Yelp, XSum, TLDR,
ELI5, WritingPrompts, ROCStories, HellaSwag, SQuAD, SciGen), 27 generators
(GPT-3.5, davinci-002/003, LLaMA 7B-65B, GLM-130B, OPT, Flan-T5, ...).
Within each domain the classes have equal counts and identical lengths.

Held-out split (1184 rows; texts under 50 words abstain, 752 scored):

| scorer | accuracy | notes |
|---|---|---|
| M5a rule-based | 0.561 | 9 thresholded rules, fully inspectable |
| M5b logistic regression | 0.689 | all 1184 rows: 0.634, ROC-AUC 0.687, CV 0.653 ± 0.015 |

M5b accuracy rises with length: 0.54 at 20-50 words, 0.64 at 50-100, 0.70
at 100-200, 0.77 at 200-300. The earlier RAID figures (0.816 / 0.847) came
from one domain (abstracts) where every human text was 96+ words, so length
alone predicted the label; they are not comparable.

## Pipeline

| module | does | public entry point |
|---|---|---|
| `m0_dataset.py` | samples + balances MAGE into CSV | `main()` |
| `m1_lexer.py` | maximal-munch lexer | `tokenize(text)` |
| `m2_symtab.py` | symbol table over the token stream | `build_symtab(tokens)` |
| `m3_parser.py` | recursive-descent CFG **and** template DFA | `parse_text(text)` |
| `m4_features.py` | 89-feature vector from M1–M3 | `extract_features(text)` |
| `m5_scorer.py` | rule scorer + trained classifier | `RuleScorer`, `ModelScorer` |
| `pipeline.py` | runs M1 to M5 on one text, stage by stage | `run_pipeline(text)` |

## Running it

```
.venv\Scripts\python.exe src\m0_dataset.py          # regenerate data/mage_clean.csv (MAGE sample)
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
- `data/` — generated `mage_clean.csv` and the `mage_pool.jsonl` fetch cache (gitignored)
- `models/` — trained classifier artifacts (gitignored, not committed; regenerate with `.venv\Scripts\python.exe src\m5_scorer.py --train`)
