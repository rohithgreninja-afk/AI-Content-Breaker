# AI Content Breaker

Compiler-design course project: an AI-generated-text detector that maps lexing,
symbol tables, CFG parsing, automata, and attribute grammars onto text analysis.
M0–M5 are implemented. M5b is a 300-tree ExtraTrees classifier over 89 lexical
and syntactic features, selected using training-only grouped validation.
Logistic regression remains available as the reference model.

## Run the React workbench

Use Node.js 22.12 or newer and Python 3.14. From the repository root, install
the Python dependencies and English POS tagger once:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m nltk.downloader averaged_perceptron_tagger_eng
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). `npm run dev` starts both
the React/Vite frontend and the Python API at `127.0.0.1:8000`; Ctrl+C stops
both, including a running training subprocess. The launcher prefers the
repository's `.venv` interpreter, then `PYTHON`, then `python`. On macOS/Linux,
use `.venv/bin/python` in the setup commands. No API keys are needed. Fonts
are bundled locally.

The workbench checks whether the saved model can actually load. **Check model**
verifies an available classifier; **Check & train model** starts background
training when one is unavailable. Training activity and failures stay visible
in the sidebar, and refreshing the page reconnects to the active job.

For analysis, paste English text or load the sample, then select **Analyze
text**. The report organizes the lexer, symbol table, parser, all 89 features,
and rule breakdown into expandable stages. **Console** contains the same
stage output as the command-line pipeline, and **Copy report** copies it.
Inputs are limited to 50,000 characters. Below 50 words both scorers abstain;
the pipeline details remain available. Submitted passages are not saved.

The included model analyzes new text without downloading the training corpus.
If the model needs training, the backend installs a missing NLTK tagger and
fetches missing pinned datasets through M0's `--parquet` path before fitting.
This first download
can take tens of minutes and needs internet access. Completed dataset blocks
are cached. Training restores the committed winning recipe and its original
training split using `src/web_training.py`; it does not run a new candidate
search or tune on the holdout. Changed datasets, runtime versions, feature
code, or uncommitted training records cause an actionable failure. Use the
explicit CLI workflow below for deliberate new model selection.

Useful frontend commands:

- `npm run dev`: start frontend and backend together.
- `npm run check`: check the React TypeScript code.
- `npm run build`: type-check and produce the frontend in `dist/`.
- `npm run preview`: preview the compiled frontend; start the Python API
  separately with `.venv\Scripts\python.exe src\web_api.py` for live analysis.

The single frontend route is `/`, with workspace and pipeline-guide anchors.
Vite forwards `/api` to the loopback-only Python server:

- `GET /api/model`: model readiness, metadata, and training activity.
- `POST /api/model/train` with `{}`: check or start one background training
  job; an already usable model is retained.
- `POST /api/analyze` with `{"text":"..."}`: structured stages, scores,
  and the original console transcript.
- `POST /api/robustness` with `{"text":"..."}`: the M6 fragility report alone.
  `POST /api/analyze` includes the same report as its last stage.
- `GET /api/sample`: the built-in demonstration passage.

POST requests use `Content-Type: application/json`. The backend validates
request size and accepts only local application hosts and origins. The React
source is in `frontend/`; the original Python CLI continues to work.

## How to train the model

Use Python 3.14. The commands below use PowerShell from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m nltk.downloader averaged_perceptron_tagger_eng
.venv\Scripts\python.exe src\m0_dataset.py
.venv\Scripts\python.exe src\m5_training.py --select
git add reports/model_selection.json
git commit -m "Lock training-only model selection"
.venv\Scripts\python.exe src\m5_training.py --finalize
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

Selection compares the eight candidates already declared in
`reports/model_selection_protocol.json`, using five-fold CV on the **saved
training groups only**. It extracts no test features during selection. The
commit freezes the winner before final fitting and evaluation; finalization
rejects uncommitted or mismatched records. This workflow requires a Git checkout.
Related questions, articles, continuations,
shared openings, and length-matched pairs stay together. Scaling is fitted
inside each training fold for candidates that need it. Boosting candidates
disable internal early stopping. The 50-word gate and 0.5 cutoff stay fixed.
The model and its estimator recipe, selection hashes, exact training/test IDs,
dataset hash, feature-extractor hash, runtime versions, and metrics are saved in
`models/m5_logreg.joblib` and `models/training_report.json`.
The artifact filename is retained for compatibility. For the legacy logistic
reference, use `src/m5_scorer.py --train --select-c` instead.

## How to do a test run of the model

The repository includes the trained classifier. After installing the
dependencies and NLTK tagger above, edit `demo_input.txt` and run:

```powershell
.venv\Scripts\python.exe demo.py --text
.venv\Scripts\python.exe src\pipeline.py path\to\your_text.txt
.venv\Scripts\python.exe src\m6_harness.py path\to\your_text.txt
```

`m6_harness.py` measures how this detector's own verdict moves. It applies
five fixed edits (synonym, transition wording, coordination, template
interruption, adjacent swap) plus one compound copy, then reports feature
deltas and whether M5a or M5b flipped. Edits that cannot apply are left out
of the fragility fraction. It is a robustness check, not an evasion tool.
The workbench shows the same report as the M6 stage after **Analyze text**.
Passages under 50 words abstain, and no flip is claimed. Rescoring uses
M1–M5 only, so the harness does not edit a passage and then measure that
edit again.

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
the full datasets or their official test sets. The current corpus has 7,490
texts (3,745 human / 3,745 AI):

- [MAGE (`yaful/MAGE`)](https://huggingface.co/datasets/yaful/MAGE): human and
  generated text across ten domains, including Reddit, reviews, summarization,
  stories, question answering, and scientific text. Source label `1` means
  human; `0` means machine. **4,734 sampled texts.**
- [HC3 (`Hello-SimpleAI/HC3`)](https://huggingface.co/datasets/Hello-SimpleAI/HC3):
  human and ChatGPT answers to the same questions; English `all` configuration.
  **1,956 sampled texts across five domains.**
- [GPT-wiki-intro (`aadityaubhat/GPT-wiki-intro`)](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro):
  Wikipedia introductions paired with `text-curie-001` introductions.
  **600 sampled texts.**
- [RAID (`liamdugan/raid`)](https://huggingface.co/datasets/liamdugan/raid):
  human and multiple-generator text, using only `attack == "none"`. The rows
  API exposes a partial training split, so observed coverage is recorded in
  the report. **200 sampled texts, all news:** the seeded pool had no clean
  human examples in its other domains, which were excluded from balancing.
  The sample does not represent all RAID domains or attacks.

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
The grouped split contains 5,580 training texts and 1,910 held-out texts.
Training-only CV selected **ExtraTrees: 300 trees, minimum leaf size 5,
70% of features per split**. Its pooled validation accuracy is **73.8% overall**
and **78.0% after the word gate**, versus **67.2% / 70.6%** for logistic C=10.
These are selection-CV results, not nested-CV estimates.

On the fixed holdout, M5b scores **73.4% accuracy / 0.821 ROC-AUC** on every
row, and **78.7% accuracy / 0.867 ROC-AUC**
on the 1,354 rows passing the 50-word gate (**70.9% coverage**, 556 abstentions).
The previously shipped logistic model scored **68.6% / 73.4%** on exactly
these all-row/gated subsets. No split, features, threshold, or gate changed.
Gated accuracy varies by source: HC3 85.1%, MAGE 75.4%, Wiki 78.1%, and RAID
67.9%; RAID has only 53 scored test rows. This holdout was published previously;
it is fixed internal evaluation, not a new unseen benchmark. The winner was
committed before this run's final evaluation and was not changed afterward.
The training score is much higher than validation/test performance, so use
the validation and holdout figures when judging the model.

`scripts/verify_training.py` checks the dataset, related-text groups, class
and length balance, and finite features, then reproduces the saved model's
complete fitted tree state and predictions (or scaler/coefficients for the
legacy linear model). It also compares combined and MAGE-only training
on identical held-out rows, writing `reports/training_validation.json`.
`models/training_report.json` preserves the previous published evaluation
separately from a fresh C=10 reference fit: the numerical solver's two-thread
refit scores 73.3% gated, slightly different from the original model's 73.4%.
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
- `src/m5_scorer.py`: `RuleScorer`, saved `ModelScorer`, and logistic baseline;
  `--audit` inspects rules on the training split.
- `src/m5_training.py`: separate training-only selection and final evaluation.
- `src/m5_estimators.py`: explicit, reproducible classifier recipes.
- `src/pipeline.py`: `run_pipeline(text)` and the test/evaluation commands.
- `tests/`: unit and regression tests; `scripts/`: exploratory data tools.
- `data/`: generated CSV, manifest, and fetch caches (ignored).
- `models/`: the trained classifier and its training report.

Each pipeline module also runs as a standalone demo. NLTK's POS tagger is
used for the shipped model. M3 can use a built-in fallback when NLTK data is
unavailable, but the model rejects a different tagger mode; install the tagger
or retrain in the intended mode. See `PROJECT_CONTEXT.md` for architecture
and `PROGRESS.md` for implementation history.
