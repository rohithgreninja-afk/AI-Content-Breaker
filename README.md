# AI Content Breaker

Compiler-design course project: an AI-generated-text detector that maps lexing,
symbol tables, CFG parsing, automata, and attribute grammars onto text analysis.
M0–M5 are implemented. M5b is logistic regression over 89 lexical and syntactic
features; additional training refits it on the combined corpus, retaining MAGE.

## How to train the model

Use Python 3.14. The commands below use PowerShell from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m nltk.downloader averaged_perceptron_tagger_eng
.venv\Scripts\python.exe src\m0_dataset.py
.venv\Scripts\python.exe src\m5_scorer.py --train --select-c
```

M0 fetches seeded samples from the four datasets listed below and writes
`data/train_clean.csv` and `data/train_manifest.json`. The first fetch can take
tens of minutes; completed blocks are cached so the same command resumes an
interrupted fetch. Every response must match its recorded dataset revision.
If the upstream revision changes, the build stops instead of silently changing
the training sample.

If the rows service repeatedly returns gateway errors, resume with
`.venv\Scripts\python.exe src\m0_dataset.py --parquet`. This reads the same
seeded row windows from separately pinned, immutable Parquet conversions;
both revision identities are recorded in the manifest and training report.

Training uses a grouped holdout and chooses regularization with five-fold CV
on the training groups only. Related questions, articles, continuations,
shared openings, and length-matched pairs stay together. Scaling is fitted
inside each training fold. The model and its exact training/test IDs, dataset
hash, feature-extractor hash, runtime versions, and metrics are saved in
`models/m5_logreg.joblib` and `models/training_report.json`.

## How to do a test run of the model

The repository includes the small trained classifier. After installing the
dependencies and NLTK tagger above, edit `demo_input.txt` and run:

```powershell
.venv\Scripts\python.exe demo.py --text
.venv\Scripts\python.exe src\pipeline.py path\to\your_text.txt
```

Use at least 50 words; the pipeline abstains on shorter inputs. A test run on
the saved model's actual held-out data requires regenerating the sampled data
with M0 first:

```powershell
.venv\Scripts\python.exe src\pipeline.py --test 0
.venv\Scripts\python.exe demo.py --eval
.venv\Scripts\python.exe -m unittest discover -s tests
.venv\Scripts\python.exe scripts\verify_training.py
```

Evaluation checks the dataset hash and saved split membership before scoring.
If you change the dataset, retrain before evaluating it. Running the pipeline
on a new text file does not require downloading the training data.

## Datasets the model was trained on

The classifier uses balanced samples of these **training splits**, rather than
the full datasets or their official test sets:

- [MAGE (`yaful/MAGE`)](https://huggingface.co/datasets/yaful/MAGE): human and
  generated text across ten domains, including Reddit, reviews, summarization,
  stories, question answering, and scientific text. Source label `1` means
  human; `0` means machine.
- [HC3 (`Hello-SimpleAI/HC3`)](https://huggingface.co/datasets/Hello-SimpleAI/HC3):
  human and ChatGPT answers to the same questions; English `all` configuration.
- [GPT-wiki-intro (`aadityaubhat/GPT-wiki-intro`)](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro):
  Wikipedia introductions paired with `text-curie-001` introductions.
- [RAID (`liamdugan/raid`)](https://huggingface.co/datasets/liamdugan/raid):
  human and multiple-generator text, using only `attack == "none"`. The rows
  API exposes a partial training split, so observed coverage is recorded in
  the report; the sample does not represent all RAID domains or attacks.

M0 balances human/AI counts within each source/domain and truncates matched
pairs to identical lengths between 20 and 300 words. It removes duplicate
texts after truncation and records connected groups to reduce leakage.
Shared 20-word openings are a conservative grouping heuristic; MAGE has no
prompt ID, so this cannot guarantee removal of every semantic overlap.
Dataset cards describe their licenses and source-specific restrictions;
downloaded source texts and caches are kept out of Git.

## Validation and limitations

The original MAGE-only classifier was audited before additional training.
Its saved coefficients and scaler match a fresh fit exactly, and its reported
accuracy (0.634 on all test rows; 0.689 on the 752 rows above the word gate)
reproduces. However, 139 original test rows share a 20-word opening with the
training set. Those historical row-split figures are not directly comparable
to the new grouped evaluation. See `reports/mage_audit.json` for the evidence.

Current training results are recorded in `models/training_report.json`.
`scripts/verify_training.py` checks the dataset, related-text groups, class
and length balance, and finite features, then reproduces the saved model's
scaler and coefficients. It also compares combined and MAGE-only training
on identical held-out rows, writing `reports/training_validation.json`.
M5a's nine inspectable rules remain frozen from the earlier MAGE training;
their new-corpus results are comparisons, because their original tuning rows
can overlap the new holdout. M5b is the newly trained classifier.

This is an experimental stylometric detector. Its scores are model outputs,
not proof of authorship. The report includes per-source results and coverage
after abstention; results on these samples do not establish performance on
new generators, adversarial text, other languages, or unseen domains.

## Pipeline and layout

- `src/m0_dataset.py`: download, clean, balance, deduplicate, and group samples.
- `src/m1_lexer.py`: maximal-munch lexer, `tokenize(text)`.
- `src/m2_symtab.py`: symbol table, `build_symtab(tokens)`.
- `src/m3_parser.py`: recursive-descent CFG and template DFA, `parse_text(text)`.
- `src/m4_features.py`: 89 features, `extract_features(text)`.
- `src/m5_scorer.py`: `RuleScorer` and `ModelScorer`; `--audit` inspects rules
  on the training split and `--train --select-c` trains the classifier.
- `src/pipeline.py`: `run_pipeline(text)` and the test/evaluation commands.
- `tests/`: unit and regression tests; `scripts/`: exploratory data tools.
- `data/`: generated CSV, manifest, and fetch caches (ignored).
- `models/`: the trained classifier and its training report.

Each pipeline module also runs as a standalone demo. NLTK's POS tagger is
used for the shipped model. M3 can use a built-in fallback when NLTK data is
unavailable, but the model rejects a different tagger mode; install the tagger
or retrain in the intended mode. See `PROJECT_CONTEXT.md` for architecture
and `PROGRESS.md` for implementation history.
