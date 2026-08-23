# AI Content Breaker - Cursor Project Context

You are helping build a compiler-design course project: an AI-generated-text
detector that reframes the compiler front-end pipeline (lexing, symbol table,
CFG parsing, finite automata, attribute grammars) as a text-analysis pipeline.
This file is the single source of truth. Obey it over your own assumptions. If
something here conflicts with what you'd normally do, follow this file and say so.

## Author / environment
- Rohith Ram V, VIT, 5th sem, Compiler Design (BCSE307P/L). Beginner-friendly.
- OS: Windows. Python 3.14.7 at .venv in repo root. Use `python -m pip`.
- Repo root: C:\Projects\AI-Content-Breaker  (git repo; keep it self-contained)
- Fully OFFLINE except the HuggingFace datasets-server API used by the RAID pull.
  No API keys exist. Do not add code that needs a paid API or secret key.

## Golden rules
- Do NOT one-shot the whole system. Build MVP-first, module by module, and STOP
  after each build step for the author to run/verify before continuing.
- After each step: print how to run it, run it yourself if possible, show real
  output, and update PROGRESS.md. Do not start the next module unprompted.
- When you hit an install failure or a missing wheel on Python 3.14, STOP and
  report it with the exact error. Never silently swap the approach.
- Diagnose bugs with a root-cause explanation before fixing. Fix one thing at a
  time. No wholesale rewrites unless asked.

## Coding style
- Readable Python. Short but MEANINGFUL names (tokens, sym_tab, feat_vec, tree),
  never single-letter compiler-lab style. snake_case functions, PascalCase classes.
- Short docstring on every module and public function. Inline `#` comments where
  logic is non-obvious. Do not over-comment trivial lines.
- No ML magic hidden in one-liners. Prefer explicit, followable code over clever.
- Keep each module in its own file under src/. No giant god-file.

## Tech stack (do not add beyond this without asking)
- Standard library + `requests` (RAID pull), `nltk` (POS tagging),
  `scikit-learn` + `numpy` (M5 classifier). `datasets`/`huggingface_hub` are
  OPTIONAL (only used by the exploratory scripts, not the pipeline).
- POS tagging: nltk `averaged_perceptron_tagger` (Penn Treebank tags). NOT spaCy
  (no reliable Python 3.14 wheel). Download nltk data on first run.
- If scikit-learn will not install on 3.14, STOP and report; do not hand-roll a
  classifier as a silent substitute.

## Dataset (M0) - already scaffolded, reuse it
- Source: `liamdugan/raid` (RAID benchmark) via HuggingFace datasets-server.
- Existing script `pull_raid.py` samples clean rows only (attack == "none"),
  balances ~1200 human / ~1300 AI, writes data\raid_clean.csv with columns:
  id, model, label, domain, text   (label in {human, ai}).
- First data task is to RUN this and confirm the CSV materializes. Fold it into
  src/ as m0_dataset.py but keep its logic. RAID is a published academic
  robustness benchmark; using a balanced clean subset to train the detector is
  its intended use.

## Architecture: 6 stages / 8 modules
Every stage is load-bearing on a specific compiler-theory concept, not a loose
metaphor. Build order follows the module numbers M0..M7.

- M0 Dataset          -> RAID clean subset -> data\raid_clean.csv
- M1 Lexer            -> maximal-munch scan into typed tokens
- M2 Symbol Table     -> word identity/frequency/positions over the token stream
- M3 Dual Syntax      -> (a) recursive-descent CFG parser -> per-sentence parse
                         trees; (b) DFA over POS-tag sequences -> template signal
- M4 Attribute Comp   -> bottom-up synthesized attributes -> FEATURE VECTOR
- M5 Hybrid Scoring   -> rule-based attr-grammar score ENSEMBLED with logistic
                         regression on the same feature vector -> verdict
- M6 Robustness Harness-> re-run M1..M5 on perturbed text, measure verdict shift
- M7 CLI / Reporting  -> single entry point: text/file in -> verdict + report out

## Interface contracts (freeze these; every module depends on them)
- Token: dict {type: "WORD"|"NUMBER"|"PUNCT"|"SENTENCE_END", value: str, idx: int}
  Token stream = list[Token]. M3 attaches pos_tag to WORD tokens.
- Symbol table: dict word(str) -> {count: int, positions: list[int]}.
- Parse tree: nested node {label: str, children: list[node]}; leaves carry the
  literal token value. One tree per sentence; parser has panic-mode recovery.
- FEATURE VECTOR (M4) - FIXED ORDER, this is the contract M5 and M6 rely on:
  [0] lexical_diversity        (type-token diversity, from M2)
  [1] burstiness               (sentence-length variance)
  [2] tree_depth_var           (parse-tree depth/branching variance, from M3a)
  [3] ngram_repetition         (repeated n-gram rate, from M3b / counts)
  [4] transition_phrase_density(high-freq/clustered entries, from M2)
  [5] firstfollow_predictability(token-bigram transition predictability)
  Return as a named structure too (dict) so features are debuggable, but the
  ordered numeric vector is what the classifier consumes.
- Verdict: {label: "human"|"ai", confidence: float(0..1), rule_score: float,
  clf_score: float, features: dict}.

## M5 scoring - staged
- Phase A (MVP): rule-based score only = hand-weighted combination of the 6
  features, framed as bottom-up synthesized-attribute evaluation. Deterministic,
  no training needed. Verdict from a threshold. Ship this first.
- Phase B: train logistic regression on RAID feature vectors, then ensemble
  rule_score and clf_score into confidence. Placeholder weights are fine until
  the dataset is materialized; mark them clearly as placeholders.

## M6 robustness harness - framing is a hard boundary
- Purpose: measure OUR OWN detector's fragility, per attribute. It is a
  robustness-evaluation tool, NOT an evasion tool. Keep that framing in code
  comments, docstrings, and the README.
- Five perturbations, each attacking one M4 attribute via the matching compiler
  mechanism, plus a compound run (all five stacked) and a fragility score
  (fraction of runs whose verdict flipped vs baseline):
  P1 lexer-aware synonym sub      -> lexical_diversity
  P2 symbol-table repetition break-> transition_phrase_density
  P3 grammar-preserving paraphrase-> tree_depth_var
  P4 POS-automaton state-break    -> ngram_repetition
  P5 bigram-entropy injection     -> firstfollow_predictability
- Report per-perturbation feature delta, confidence delta, and flipped Y/N.

## Repo layout to create
```
src/  m0_dataset.py m1_lexer.py m2_symtab.py m3_syntax.py
      m4_attributes.py m5_scoring.py m6_harness.py m7_cli.py
tests/ smoke tests per module
data/ raid_clean.csv (generated)
models/ trained classifier (generated)
PROGRESS.md  README.md  requirements.txt  PROJECT_CONTEXT.md (this file)
```
Keep the existing exploratory scripts (pull_raid.py etc.) at root or move to a
scripts/ folder; do not delete them.

## Reporting back
Maintain PROGRESS.md: what each step built, how to run it, what was verified,
and what's next. This project is graded on showing intermediate steps with
justification, so keep the reasoning visible, not just final tables.
