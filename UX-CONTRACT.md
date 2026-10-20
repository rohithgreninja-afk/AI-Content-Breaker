# Workbench interaction contract

This is one local analysis workflow; there are no sibling screens or pre-existing UI owners.

| Capability | Canonical owner | Source of truth | Variants | Verification |
| --- | --- | --- | --- | --- |
| Form | frontend/App.tsx analysis form | Pipeline word gate; API character limit | Empty, editing, analyzing, error, complete | Live browser submission and validation |
| Scrollbar | frontend/styles.css global baseline | DESIGN.md CSS tokens | Document, textarea, console | Desktop and narrow viewport inspection |
| Feedback | frontend/App.tsx inline status regions | API response | Status, error, training | Live browser and API tests |
| Report navigation | frontend/App.tsx report links and native details | Pipeline stages | Structured report, original console | Keyboard and browser checks |

The analysis form uses a labeled textarea and app-owned validation with noValidate. Empty submission focuses the field; API errors preserve text. Text is transient in React memory and never placed in URLs or local storage. At fewer than 50 words analysis remains available, but both verdicts abstain. Editing after analysis marks the report as belonging to the previous input until reanalysis.

Check model checks actual loadability. A missing/unusable model starts one background training task; repeated requests reuse it. Polling stops at terminal status. The action never replaces a working model. Training errors remain visible with a retry action.

All output returned by the CLI is available verbatim in the console view and organized by actual M1–M5 stages in the pipeline view. The trained-model and rule-based scores are separate; neither is described as confidence or proof.

Buttons use native semantics, visible keyboard focus, stable busy labels, hover and disabled states. Disclosures use native details/summary. No custom select, dates, modal, toast, CRUD, or row selection is needed. Responsive layout stacks the input before results, permits internal code scrolling, and prevents page-wide overflow. Motion honors reduced-motion.
