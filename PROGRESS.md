# PROGRESS

## 2026-10-26: M8 keeps lowering the score

- A human verdict no longer ends a translation pass. Each call applies one
  rewrite that lowers the M5b score, and the next call continues from that
  text. The workbench **Humanize text** button sends the latest rewrite
  when the source field has not been edited.

## 2026-10-25: M8 documentation

- README now opens with M0–M7 as the detector and M8 as the separate
  translation pass. PRODUCT, DESIGN, the workbench direction, and the design
  catalog describe **Humanize text** and the M8 block. The rail, sidebar, and
  pipeline guide remain M1–M7.

## 2026-10-25: M8 attribute-directed translation

- Added `src/m8_humanize.py`. A fired M5a rule selects one rewrite from a
  fixed table, then M1–M5 score the passage again. M5b is recorded and does
  not choose the edit. M8 is not in `STAGES` or `CORE_STAGES`, so Analyze
  and the M6 harness still score the original text.
- The pass order is coordinated split, subordinator split, commas before
  bare coordinators, synonym collapse, then a later-repeat replacement.
  `pos_verb_rate` has no meaning-preserving rewrite and is named when it
  remains fired. The loop stops when both verdicts are human, no fired rule
  still has a rewrite, or 6 passes have run. Under 50 words the text is
  unchanged.
- `POST /api/humanize` uses the same local-origin and size checks as
  analysis. The workbench **Humanize text** button shows the passes, the
  rewritten passage, **Copy text**, and **Use in editor**. The M1–M7 rail
  is unchanged.
- Run: `.venv\Scripts\python.exe src\m8_humanize.py`
- Tests: `tests/test_m8_humanize.py` (15 tests) and the humanize cases in
  `tests/test_web_api.py`. On the built-in sample the pass printed
  `result: passed`, with M5a moving from 0.560 ai to 0.480 human and M5b
  from 0.552 ai to 0.407 human, after four rewrites. The saved model file
  was not rewritten.

## 2026-10-25: M7 reporting entry

- Added `src/m7_cli.py`. A passage, a file, or the built-in sample goes in.
  The verdict block comes out first, then the existing M1–M6 transcript.
  `demo.py --text` calls this entry. `--test` and `--eval` still forward to
  `pipeline.main`.
- M7 is not in `STAGES` or `CORE_STAGES`. M6 still rescores through M1–M5
  only. The report keeps M5a (`rule_score`) and M5b (`clf_score`) separate
  and does not call either a confidence. Under 50 words both verdicts abstain.
- The workbench labels that same block **M7 · Report**. The rail and the
  pipeline guide include M7 and scroll to it. Stage disclosures stay M1–M6.
  Console and Copy report start with the two verdict lines.
- Run: `.venv\Scripts\python.exe src\m7_cli.py`
- Tests: `tests/test_m7_cli.py` (4 tests). A bare
  `src\m7_cli.py` on the built-in sample printed `M5a ai` at 0.560 and
  `M5b ai` at 0.552, then every stage heading from M1 through M6. Fragility
  on that run was M5a 0/6 and M5b 1/6. The saved model file was not rewritten.

## 2026-10-25: M6 robustness harness

- Added `src/m6_harness.py` and the pipeline row
  `("M6 robustness", run_m6, show_m6)`. `CORE_STAGES` stays M1–M5, and the
  harness rescores through that list so an edit cannot recurse into M6.
  Ordinary analysis now runs M1–M6.
- The edits map onto features that already exist: synonym substitution
  (`lex_type_token_ratio`), transition replacement (`lex_transition_rate`),
  coordination rewrite (`parse_stdev_depth`), template interruption
  (`tpl_score`), and an adjacent content-word swap (`rep_word_entropy`).
  Unapplied edits are left out of the fragility fraction. Under 50 words the
  baseline abstains and no flip is claimed. M5a and M5b are reported separately.
- The edits are a fixed table.
- `POST /api/robustness` returns the same report. The workbench renders it
  as the M6 stage of the analysis, in the rail and the pipeline guide.
- Run: `.venv\Scripts\python.exe src\m6_harness.py path\to\text.txt`
- Tests: `tests/test_m6_harness.py` and the robustness cases in
  `tests/test_web_api.py`. Analyze on a long passage prints M1–M6, and the
  M6 stage is the six-row fragility report.

## 2026-10-20: React analysis workbench

- Added a React/TypeScript frontend in `frontend/` with a focused source editor,
  model-readiness controls, live word count, sample text, and separate structured
  pipeline and original-console reports. M1-M5 remain inspectable, including
  token types, vocabulary frequencies, sentence tags/trees, 89 features, and
  fired rules. Copying a report preserves the CLI transcript.
- `npm run dev` starts Vite on `127.0.0.1:5173` and the standard-library Python
  API on `127.0.0.1:8000`. The runner selects the local `.venv` first, checks
  Python dependencies, forwards `/api`, and stops both process trees on Ctrl+C.
  React dependencies and local fonts are locked in `package-lock.json`.
- Added `src/web_api.py` with model status, asynchronous train/check, sample,
  and analysis endpoints. Requests validate local hosts/origins, JSON type,
  payload size, and the 50,000-character limit. Analysis does not save input
  text and retains the existing 50-word abstention gate.
- Training checks whether the model actually loads, retains a ready model,
  prevents duplicate jobs, and reports activity/failures while the interface
  stays usable. `src/web_training.py` can recreate an unavailable artifact from
  committed protocol/selection/report records and the original training IDs.
  It validates runtime, dataset, manifest, feature and cache identities, fits
  the frozen winner, and does not perform candidate search or holdout tuning.
  Missing tagger/data downloads use the existing project tools; initial corpus
  acquisition needs internet and can take tens of minutes.
- Documented Python/Node setup, `npm install`, `npm run dev`, production build,
  API contracts, and recovery constraints in README. Existing CLI entry points,
  trained-model provenance, and model-selection workflow remain available.
- Verification: `npm run build` passes TypeScript and Vite production bundling.
  A real `npm run dev` boot returned HTTP 200 from the frontend and proxied
  model status (`ready: true`, ExtraTrees); Ctrl+C released both listening ports.
  `.venv\Scripts\python.exe -m unittest discover -s tests` passes all **197
  tests** in 156.6 seconds, including CLI/API output parity, malformed requests,
  local-origin protection, duplicate-job prevention, training failure recovery,
  and a real temporary-artifact refit reproducing the shipped model's scores.
- Browser verification covered the empty form, model readiness, sample loading,
  complete pipeline/console output, tokens, first-sentence tree, all features,
  short-text abstention, stale input, and recovery after stopping the server.
  Checked 1440px, 887px, and 390px layouts with no document overflow. Simplified
  the copy following user feedback and increased feature/table data to 12px
  after an independent review; the reviewer scored that fix resolved.
- Missing-manifest retries now regenerate the pinned dataset, and normal backend
  shutdown stops its owned training process. The console transcript includes
  the CLI-style input header. The final focused backend run passes all 16 tests.
  Static UI audit has zero findings; design documentation lint has zero errors.

## 2026-10-20: Improve accuracy using training-only model selection (complete)

This section supersedes the classifier choice and metrics in the previous
training entry. The corpus, feature extractor, split, gate, and cutoff are
unchanged.

- Committed `reports/model_selection_protocol.json` before trials: eight
  explicit candidates, five fixed StratifiedGroupKFold folds (seed 42),
  pooled gated validation accuracy as the primary metric, all-row accuracy
  as tie-breaker/guard, minimum improvement 1 percentage point. The gate
  remains 50 alphabetic whitespace pieces and the probability cutoff 0.5.
  Boosting candidates explicitly disable internal early-stopping validation.
- `m5_training.py --select` builds a separate, hashed feature cache containing
  **only the original 5,580 training IDs**, never the 1,910 heldout texts.
  All learned preprocessing is fitted inside each fold. Related documents
  stay together; all candidates see identical folds. There is no rule-score
  blending, threshold tuning, or test-result-driven candidate selection.
- ExtraTrees won the frozen rule: **0.7382 all-row / 0.7797 gated pooled OOF
  accuracy**, versus logistic C=10 **0.6720 / 0.7061**. These are selection-CV
  results, not unbiased nested-CV estimates. Candidate details and each
  training ID's fold are saved in `reports/model_selection.json`.
- Committed the winner in `52d0d17` before fitting or evaluating it on test
  data. Finalization checks committed protocol/selection content, cache
  identity and digest, unchanged split IDs, complete fold coverage, and
  related-group consistency before any heldout feature extraction. It fits
  only the frozen recipe and never switches models based on the final test.
- The saved classifier is **ExtraTrees, 300 trees, minimum leaf size 5,
  max_features=0.7, random_state=42**. It completes the fixed tree budget.
  Same 89 features and saved split: 5,580 training / 1,910 test texts;
  1,354 pass the word gate, 556 abstain (**70.9% coverage**).
- **Fixed internal holdout:** all-row accuracy **0.7335**, ROC-AUC **0.8213**;
  gated accuracy **0.7873**, ROC-AUC **0.8672**, precision **0.8174**, recall
  **0.7430**, F1 **0.7785**. Gated confusion `[[560,113],[175,506]]`.
  The previously shipped model scored **0.6864 / 0.7341** all-row/gated;
  the gain is **4.7 / 5.3 percentage points** with unchanged subsets.
- Gated accuracy by source: MAGE **0.7544** (688 rows), HC3 **0.8506**
  (462), Wiki **0.7815** (151), RAID **0.6792** (53). This is the already
  published internal holdout, not a new unseen external benchmark. Training
  accuracy is 0.9975, far above validation/test; do not treat it as a
  generalization estimate. No further search followed the final test results.
- Preserved the exact previous published evaluation separately from the
  C=10 reference refit. The two-thread refit's numerical optimizer path
  scores 0.7326 gated, slightly different from the original artifact's
  0.7341. Verified prior/current dataset, manifest, feature hashes, train/test
  IDs and order are identical; the original model hash is retained.
- `ModelScorer` still loads the original logistic artifact and accepts new
  sklearn recipes, validating saved constructor parameters against the
  recorded recipe. Nonlinear models refuse to present signed linear
  coefficients. Batch evaluation and bounded prediction threads prevent
  needless parallel overhead; threadpoolctl is pinned to the used version.
- `verify_training.py` now checks complete fitted tree state or legacy
  linear parameters as appropriate. A fresh fit reproduces every tree node
  and value exactly (**maximum difference 0.0**); probability differences
  are only **2.22e-16**. Recorded holdout metrics match actual saved-model
  predictions; features are finite and there are no duplicate or leaked
  groups. Evidence: `reports/training_validation.json`.
- **All 181 tests pass**, including 23 selection/finalization leakage tests
  and six added model/verification tests. Sentinel texts prove excluded
  heldout rows never reach selection extraction. Failed commit/cache/fold
  checks stop before heldout access. `pip check` is clean and the demo runs.
- README now documents the selection/commit/finalization workflow, current
  classifier and results, compatibility artifact filename, and evaluation
  limitations. The approximately 34 MB artifact is included; the corpus and
  feature cache remain ignored. No new datasets or feature changes were
  needed for this improvement.

## 2026-10-20: MAGE audit and additional dataset training (complete)

This section supersedes the historical RAID and MAGE scoreboards below.

- Audited the original MAGE model before replacing it. Its saved scaler and
  coefficients reproduce exactly, as do the README metrics (0.634 all-row
  accuracy; 0.689 after the word gate). The original row split allowed 139
  test texts to share a 20-word opening with training texts; the new training
  uses connected document groups instead. Evidence: `reports/mage_audit.json`.
- Finished M0's combined corpus: **7,490 texts, 3,745 per class**. MAGE 4,734;
  HC3 1,956 across five English domains; GPT-wiki-intro 600; RAID 200 news
  texts. RAID's API/conversion exposes only a partial split, and the seeded
  pool had no clean human rows in the other available domains. Those domains
  contribute no training examples. No all-domain RAID claim is made.
- Every source/domain has identical human/AI word-length histograms, with
  20–300 words per text. Removed one duplicate pair after truncation. Grouped
  related source IDs, shared 20-word openings across corpora, and matched
  pairs; 2,796 independent groups remain. No normalized duplicate text or
  related group crosses the train/test boundary.
- Fetching verifies source revisions and checkpoints complete blocks, recovers
  interrupted JSONL tails, rejects truncated API cells, and uses canonical
  sorted pool digests. The rows service repeatedly returned 502/503, so the
  completed run used `--parquet`: bounded byte ranges from separately pinned
  immutable conversions. Every original HC3 API cache row (1,315 unique
  rows) matches Parquet in every field; Wiki/RAID representative rows also
  match. Evidence and verification limits: `reports/dataset_transport_audit.json`.
- **M5b retrained and saved** on 5,580 texts (2,130 groups), holding out 1,910
  texts (666 groups). Every dataset contributes training rows: MAGE 3,642;
  HC3 1,352; Wiki 440; RAID 146. Scaling stays inside the training folds.
  Five-fold grouped training-only CV selected **C=10**, accuracy
  **0.672 ± 0.019**. Logistic regression converged in **247/2,000 iterations**.
- Held-out model accuracy **0.6864**, ROC-AUC **0.7550**; confusion
  `[[668,287],[312,643]]`. The pipeline abstains below 50 alphabetic words:
  556 abstentions, 1,354 scored rows (**70.9% coverage**), accuracy **0.7341**.
  Source-specific gated accuracies: HC3 0.8485; MAGE 0.6846; Wiki 0.6623;
  RAID 0.5849 (only 53 scored RAID rows).
- M5a's nine rules stay frozen as an inspectable MAGE comparison. Their new
  corpus metrics are descriptive, since their historical tuning rows can
  overlap this holdout; they are not an independent rule benchmark.
- The classifier now stores dataset/manifest/feature hashes, exact split IDs
  and groups, tagger mode, and runtime versions. Evaluation refuses mismatched
  data, altered groups, or stale provenance, and scores with the validated
  model instance. Model loading rejects a changed feature extractor or tagger.
- Shipped the small `models/m5_logreg.joblib` and `models/training_report.json`.
  Downloaded corpora/caches remain ignored. Pinned training dependencies.
  README now begins with training commands, test runs, and all four datasets.
- **Verified:** all **152 tests pass**, including the actual saved holdout;
  `pip check` is clean; demo text and heldout-row commands work. A cached M0
  rebuild reproduces the exact CSV and manifest hashes. `verify_training.py`
  confirms finite features, balance, no duplicate/leaked texts, convergence,
  and model/report metric agreement. A fresh fit reproduces all scaler means,
  scales, coefficients, and intercepts with maximum difference **0.0**.
- On the identical final gated holdout, a MAGE-only reference fit (C=30)
  scores **0.6773**, versus **0.7341** for combined training. Broader training
  improves the overall reference comparison but reduces MAGE-only accuracy
  from **0.7137 to 0.6846**. This is an internal sample comparison, not an
  external benchmark. Evidence: `reports/training_validation.json`.

Reproduce with the PowerShell commands near the beginning of `README.md`.
For outage recovery use `src/m0_dataset.py --parquet`; train with
`src/m5_scorer.py --train --select-c`; verify with `scripts/verify_training.py`.

## Historical implementation notes

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

## Repo hygiene (done)
- Removed `.cursorrules` from git tracking (`git rm --cached`); file stays
  locally for Cursor but is no longer pushed to GitHub.
- Added `.cursor/` and `.cursorrules` to `.gitignore` so they can't be
  re-added accidentally.
- Committed and pushed as `chore: remove cursor rules from tracking, update gitignore`.
- Tracked files are now code + docs only; no IDE artefacts in the remote.

## M3 - Dual syntax (DONE, verified)
- `src/m3_parser.py`: `parse_text(text) -> dict` and `parse_tokens(tokens) -> dict`
  (the second so M4 can tokenize once and reuse the stream).
- **POS tagging:** nltk `pos_tag` on pre-tokenized M1 WORD/NUMBER values, so no
  `punkt` is needed. Penn tags are coarsened to
  `NOUN VERB MODAL ADJ ADV DET PRON PREP CONJ PART NUM INTJ X PUNCT END`.
  `PRP$`/`WP$` map to DET (they fill the determiner slot).
  - Tagger data is **not** in the repo. First run downloads
    `averaged_perceptron_tagger_eng` (nltk >= 3.9 renamed it; both names tried).
    If that fails, a deterministic closed-class + suffix fallback tagger runs so
    a fresh clone still works offline. Mode is reported as `meta["tagger"]`.
- **(a) Recursive-descent CFG** over coarse tags, grammar in the module
  docstring. Emits nested `{label, children}` trees -> depth, nodes, branching,
  clause/PP/subordinate counts, `parse_ok`, `coverage`.
  - `PUNCT` is a real terminal admitted only at phrase/clause boundaries. This
    is what stops a VP's complement list from reading across a comma
    ("Although the results improve, we note X" splits where it should).
  - Penn tags are threaded in purely to gate verb chaining: only `VB/VBN/VBG`
    continue an auxiliary chain, so "works is" (VBZ VBZ) is two predicates.
    Without this a relative clause ate the main verb and the sentence failed.
  - A leading `DET` is **not** a subordinator; allowing it made "The cat sat"
    parse as a subordinate clause with no main clause (success rate 0.000).
  - Participles do double duty: premodifier ("the proposed method") and reduced
    relative ("images using a graph"). Both need the Penn tag since VBN/VBG
    coarsen to VERB.
  - `MAX_CLAUSE_DEPTH = 40` caps nesting so a pathological row cannot blow the
    Python stack. Verified: no crash or hang on 600-word runs, 200 consecutive
    commas, 150 chained conjunctions, unicode, emoji, empty.
- **(b) Template DFA:** `build_dfa` compiles tag patterns into a deterministic
  prefix automaton (shared prefixes collapse into shared states), explicit
  `{state: {tag: state}}` table, longest match wins. Two automata: one anchored
  at sentence start, one run at every offset. 8 template classes.
- **Measured on all 760 rows** (7 ms/row, 5.0s total, tagger=nltk):

  | metric | human | ai | delta |
  |---|---|---|---|
  | parse_success_rate | 0.233 +- 0.182 | 0.270 +- 0.247 | +0.036 |
  | mean_coverage | 0.494 +- 0.174 | 0.534 +- 0.208 | +0.041 |
  | mean_depth | 6.58 +- 2.23 | 8.10 +- 2.98 | +1.53 |
  | mean_branching | 1.48 +- 0.42 | 1.65 +- 0.52 | +0.17 |
  | mean_subordinate | 0.152 +- 0.166 | 0.216 +- 0.230 | +0.064 |
  | template_score | 0.316 +- 0.181 | 0.303 +- 0.238 | -0.014 |

  Strongest single template: `nominalization_chain` (human 0.130, ai 0.400).
  No zero-variance metric, so every feature carries signal into M4.
- **Known limitation:** 72.5% of sentences are partial parses corpus-wide. The
  remaining halts are finite verbs (VBZ 350) and appositive nouns (NN 268) where
  an earlier NP over- or under-consumed. Fixing those needs backtracking or a
  chart parser (Earley/CYK) - a different architecture, deliberately not built.
  `parse_success_rate` and `coverage` are therefore *partial-parse* signals,
  which is fine: they vary and they separate the classes.
- Tests: 33 cases in `tests/test_m3_parser.py` (all passed).
- Run demo: `.venv\Scripts\python.exe src\m3_parser.py`
- Run tests: `.venv\Scripts\python.exe tests\test_m3_parser.py`

## M4 - Feature extractor (DONE, verified)
- `src/m4_features.py`: `extract_features(text) -> dict[str, float]`, plus
  `extract_features_from(tokens, sym_tab, meta)` when the caller already holds
  the M1/M2/M3 artefacts (M5 uses it to avoid re-parsing).
- **89 features** in 7 groups, keyed by prefix:
  `lex_` (14) token mix, word length, richness, function/transition/hedge/
  booster rates; `rep_` (10) hapax, top-word mass, repeat gaps, word entropy;
  `sent_` (10) length mean/stdev/CV/range and burstiness; `pos_` (19) coarse
  POS shares plus noun:verb, adj:noun, adv:verb ratios; `parse_` (11) depth,
  branching, nodes, clauses, PPs, subordination, success/coverage;
  `tpl_` (14) DFA template rates, structural mix, distinct-template count;
  `punct_` (11) per-mark rates.
- **Contract M5 relies on:** `FEATURE_NAMES` is the canonical order and
  `N_FEATURES` its length. Every text yields every key (0.0 when absent), so
  the design matrix is always rectangular. All values are finite floats.
  - `FEATURE_NAMES` is derived at import from a probe text, so the published
    order can never drift out of step with the functions that build the dict.
  - Everything is a rate, share, or mean; raw counts appear only as `log1p`,
    so features stay comparable across texts of different lengths.
  - All division goes through `_ratio`, which returns 0.0 on an empty
    denominator. scikit-learn rejects NaN at fit time, so this matters.
- **Verified on all 760 rows** (5 ms/row, 3.8s total): 0 non-finite values,
  0 zero-variance features, 32 features with |Cohen's d| >= 0.2.
  Strongest signals (+ = higher in AI):

  | feature | d | human | ai |
  |---|---|---|---|
  | parse_mean_depth | +0.580 | 6.58 | 8.10 |
  | punct_dash_rate | -0.571 | 0.0279 | 0.0173 |
  | pos_punct_rate | -0.477 | 0.1144 | 0.0803 |
  | parse_mean_nodes | +0.434 | 19.84 | 23.59 |
  | rep_dis_rate | -0.434 | 0.1454 | 0.1149 |
  | rep_stdev_repeat_gap_log | -0.418 | 3.602 | 3.337 |

  Notably the best single feature is an M3 parse feature, so the compiler
  front end is carrying real signal rather than decorating a bag of words.
  Humans punctuate more (dashes, commas, parens); AI nests deeper.
- Tests: 18 cases in `tests/test_m4_features.py` (all passed), covering the
  fixed-width/order contract, finiteness on degenerate input, and spot checks
  that POS and structure shares each sum to 1.0.
- Run demo: `.venv\Scripts\python.exe src\m4_features.py`
- Run tests: `.venv\Scripts\python.exe tests\test_m4_features.py`

## M5 - Scorer + classifier (DONE, verified)
- `src/m5_scorer.py`. Both scorers share one interface: text -> probability in
  0.0-1.0 that the text is machine-generated. `AI_LABEL = "ai"` is always the
  positive class (1), whatever order sklearn would otherwise infer.

### M5a RuleScorer (rule-based, no training)
- `RULES` is a plain table of `(feature, direction, threshold, weight)`. A rule
  that fires adds its weight, one that misses subtracts it; the signed total
  goes through a logistic so the output reads as a probability.
- `explain(features)` returns every rule's contribution, so a score is always
  inspectable - that is the point of M5a over M5b.
- **18 rules.** Thresholds are the cut points that maximised *lift* (AI hit
  rate minus human hit rate) on the sample, constrained to fire on 15-95% of
  AI rows. Weights follow lift in tiers (>=0.30 -> 1.6 down to <0.10 -> 0.4).
- **Two candidate rules were dropped, not retuned.**
  `tpl_topic_sentence_rate` and `lex_transition_rate` both had their best cut
  in the *opposite* direction to the hypothesis (AI using more topic sentences
  and connectives), at lift <= 0.10. In this corpus AI uses *fewer* explicit
  connectives than humans. Flipping a rule to chase a sign with no story
  behind it is curve-fitting, so they are out rather than inverted.
- **`RULE_BIAS` matters more than it looks.** At bias 0 a text must fire more
  than half the total weight to clear the 0.5 cutoff, and most texts fire
  fewer, so the signed total was negative for *both* classes (mean score:
  human 0.08, ai 0.42) and the scorer called nearly everything human
  (recall 0.41). Re-centring to **+4.5** moved accuracy 0.705 -> 0.774 and
  recall -> 0.63.
- **`RULE_SCALE` cannot change accuracy** - any scale > 0 is monotonic through
  a fixed 0.5 cutoff, so the labels never move. It is a calibration knob only,
  picked on Brier score (**0.20**, Brier 0.161). `calibrate_rules` grids bias
  on accuracy first, then scale on Brier, in that order for that reason.
  - Edge case: when the total exactly cancels the bias, the score sits ~1e-16
    from the boundary and float rounding inside `exp()` picks the side. That
    tie is inherent to any squashing function; covered by a test.
- Results: **in-sample 0.774** / **held-out 0.816** accuracy (p 0.877,
  r 0.711, f1 0.785), separation +0.361. Thresholds were tuned on all 760
  rows, so the in-sample figure is optimistic; the held-out number shares
  M5b's split and is the one to compare.

### M5b ModelScorer (logistic regression)
- `StandardScaler` -> `LogisticRegression` pipeline. Scaling is required: the
  89-feature vector mixes 0-1 rates with log counts and raw tree depths.
- **C = 30**, chosen by `select_C()`: 5-fold CV on the **training split only**,
  never the test split. Train CV accuracy climbs 0.753 (C=0.01) -> 0.867 and
  plateaus at 30. Stronger regularisation underfits here.
- `fit()` runs its reported CV on the training split only too. Running it over
  train+test would leak test rows into a number printed beside test accuracy.
- Results: **test accuracy 0.847**, **ROC-AUC 0.941**, train CV 0.867 +- 0.031,
  confusion `[[86, 14], [15, 75]]`. Train 0.951 vs test 0.847 is a real gap,
  but CV tracks test and both improve with C, so the features support the
  flexible boundary rather than memorising.
- `save`/`load` persist to `models/m5_logreg.joblib` (gitignored, regenerable).
  `load` **rejects a model whose feature order does not match M4's
  FEATURE_NAMES** - a stale model silently scoring a reordered vector would be
  the worst kind of bug here.
- Top standardised coefficients are a correlated vocabulary-richness cluster
  (`rep_word_entropy` -18.9, `lex_type_token_ratio` +16.9, `rep_hapax_rate`
  -8.4). Collinearity inflates individual magnitudes, so read them as a group,
  not as independent importances. `parse_mean_depth` (+2.7) carries M3's signal.

### Scoreboard (RAID era, superseded; see "Dataset switch: RAID -> MAGE")
| scorer | RAID accuracy | MAGE accuracy (752 rows over 50 words) |
|---|---|---|
| M5a rules | 0.816 held-out | 0.561 |
| M5b logreg | 0.847 test | 0.689 (0.634 on all 1184; AUC 0.687) |

- CLI: `src\m5_scorer.py` (report), `--audit` (per-rule lift + calibration
  grids, for tuning), `--train` (fit, report, save), `--score FILE` (score one
  text with both, with the rule breakdown).
- Tests: 30 cases in `tests/test_m5_scorer.py` (all passed).
- Run: `.venv\Scripts\python.exe src\m5_scorer.py --train`
- Run tests: `.venv\Scripts\python.exe tests\test_m5_scorer.py`

## Dataset switch: RAID -> MAGE (done)
- **Why:** the RAID sample was 100% scientific abstracts from 6 older
  models, and every human row was 96+ words. A length-only tree scored 0.80,
  so the reported 0.816 / 0.847 mostly measured length and genre.
- **Source:** `yaful/MAGE` (pinned revision 342663f0), train split, 319k rows.
  The CDN served ~22 KB/s here, so M0 samples through the HF datasets-server
  rows API: 200 blocks of 100 rows at seeded random offsets, 1 s apart with
  backoff on 429/502, cached and resumable in `data/mage_pool.jsonl`
  (19,187 rows). The `/filter` endpoint never finished loading its index.
- **Balancing:** per domain, equal human/AI counts (max 300 pairs); each
  human row is paired with an AI row and both are cut to the shorter length
  (20-300 words). Human and AI length distributions are identical (median
  67). Result: 4734 rows, 10 domains, 27 generators. MAGE labels (1 = human)
  are mapped to the human/ai contract; LLaMA rows (tagged by size) are named
  `llama_<size>`.
- **M5a re-derived** on the train split only (it was fitted on all RAID rows
  before): 9 of 18 rules survive, all at weight 0.4 (best lift 0.09);
  RULE_BIAS 4.5 -> 0.0. Held-out 0.523 (RAID rules) -> 0.531.
- **M5b retrained** (C=30 still best by CV): test 0.634, AUC 0.687, CV
  0.653 +- 0.015. By length: 0.54 / 0.64 / 0.70 / 0.77 at 20-50 / 50-100 /
  100-200 / 200-300 words. Hardest generators: llama_65B (recall 0.14),
  text-davinci-003 (0.39), gpt-3.5 (0.43). HistGradientBoosting on the same
  features reaches 0.702 (not adopted; logistic regression is the spec).
- `--eval`: 432 of 1184 test rows abstain (under 50 words); on the other 752
  M5a 0.561, M5b 0.689.
- **Reading the numbers:** these features are lexical/syntactic only (no
  language model), and MAGE spans genres and modern LLMs, so ~0.65-0.70 is
  the honest ceiling for this design, not a regression.

## Short-text abstain gate (pipeline, done)
- **Bug:** short text was called AI. Root cause: every human training row is
  96+ words (all RAID abstracts), and 7 M5a "low" rules fire when their
  feature is 0.0 just because the text is short (no repeats, one sentence, no
  punctuation). That gives 10.2 of 20.5 weight for free, against a break-even
  of 8.0 set by RULE_BIAS. Human abstracts cut to 20 words: M5a calls 75% AI,
  M5b 99%. No bias retune fixes it: at 30 words or fewer neither scorer beats
  chance.
- **Fix:** `pipeline.MIN_WORDS = 50` replaces the print-only `MIN_WORDS_NOTE =
  20`. Below it `run_m5` sets `ctx["abstain"] = True` and
  `ctx["verdicts"] = {"M5a": "abstain", "M5b": "abstain"}` (M5b is `None` when
  no model is saved); `show_m5` prints no score; `--eval` counts abstained rows
  and leaves them out of accuracy. Scores are still computed into ctx for
  inspection, so read `ctx["verdicts"]`, not the raw score, for a label.
- **Word unit:** `ctx["n_words"]` counts whitespace-separated pieces that hold
  a letter, the unit the 96-word human minimum was measured in. M1 tokens were
  tried first and rejected: a 2-word markdown link with a hex hash counted as
  77 words and got a verdict.
- `--eval` now (nltk tagger): 4 of 190 rows abstain (all AI); M5a 0.742 and
  M5b 0.849 on the other 186. Human rows are unchanged ([[86, 14]] for both).
- The built-in `SAMPLE` was lengthened to 53 words so a bare
  `python src/pipeline.py` still shows a verdict.
- **Not covered:** `m5_scorer.py --score FILE` and `RuleScorer/ModelScorer
  .predict()` have no gate. **50-100 words is still weak**, because no human
  training text is that short. Held-out human abstracts cut to 50 / 70 / 90
  words: M5b calls 89% / 70% / 49% AI and M5a 40% / 36% / 29%, against 14% at
  full length. A gate of 100 would remove most of that band but abstain on 16%
  of full-length AI rows. The real fix is training on short windows.
- **Environment trap seen while testing:** in the Claude desktop shell nltk
  rejected the virtualised AppData path (`pathsec ... Unauthorized path`), and
  `tagger_mode()` silently fell back. M5a recall dropped to 0.37 and M5b to
  0.75. Copying `taggers/` into `.venv\nltk_data` fixed it. Check
  `meta["tagger"] == "nltk"` before trusting any accuracy figure.

## Test suite

The early M1–M5 smoke files were 92 tests (M1 6, M2 5, M3 33, M4 18, M5 30).
That table went stale once the pipeline, dataset, training, and workbench
suites landed: the 2026-10-20 entry records **197** tests. M6 adds
`tests/test_m6_harness.py` and the robustness cases in `tests/test_web_api.py`.

## Caveats worth carrying forward
- The 760-row sample is small and only 6 AI models wide; it is `attack == none`
  only, so nothing here is tested against adversarial/paraphrased text.
- M5a thresholds and M5b's C were both tuned on this sample. Re-run `--audit`
  and `select_C()` after regenerating the dataset.
- 72.5% of sentences are partial parses (see M3), so `parse_*` features are
  partial-parse signals. They separate the classes anyway.
