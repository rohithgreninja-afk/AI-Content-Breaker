# Workbench interaction contract

This is one local analysis workflow; there are no sibling screens or pre-existing UI owners.

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
| --- | --- | --- | --- | --- |
| Form | frontend/App.tsx analysis form | Pipeline word gate; API character limit | Empty, editing, analyzing, error, complete | Live browser submission and validation |
| Scrollbar | frontend/styles.css global baseline | DESIGN.md CSS tokens | Document, textarea, console | Desktop and narrow viewport inspection |
| Feedback | frontend/App.tsx inline status regions | API response | Status, error, training | Live browser and API tests |
| Report navigation | frontend/App.tsx report links and native details | M1–M6 stage disclosures; M7 verdict block | Structured stages, verdict-first console | Keyboard and browser checks |
| Humanize | frontend/App.tsx Humanize text button and M8 block | M5a fired rules; 50-word gate | Rewritten passage, copy, use in editor | Live browser humanize, then analyze the loaded rewrite |

The analysis form uses a labeled textarea and app-owned validation with noValidate. Empty submission focuses the field; API errors preserve text. Text is transient in React memory and never placed in URLs or local storage. At fewer than 50 words analysis remains available, but both verdicts abstain. Editing after analysis marks the report as belonging to the previous input until reanalysis.

Check model checks actual loadability. A missing/unusable model starts one background training task; repeated requests reuse it. Polling stops at terminal status. The action never replaces a working model. Training errors remain visible with a retry action.

The console view and Copy report show the M7 transcript: both verdicts, then the M1–M6 stage text. The pipeline view organizes M1–M6 in disclosures. The M7 rail link scrolls to the verdict block and does not add a seventh disclosure. The trained-model and rule-based scores are separate; neither is described as confidence or proof.

Humanize text is a separate action. It posts to `/api/humanize` and replaces the results panel with the M8 translation block. A native select switches between the before humanizing report and the after humanizing report. Analyze replaces that panel with the M7 report. The two reports are not shown together. Each humanize report shows that side’s verdicts, scores, and passage. The source field changes only when Use in editor is selected on the after report. Editing after a rewrite marks that block as belonging to the previous input. The M1–M7 rail stays the detector rail. A human verdict after rewriting is a detector output.

Buttons use native semantics, visible keyboard focus, stable busy labels, hover and disabled states. Disclosures use native details/summary. The humanize report switch is a native select. No custom select, dates, modal, toast, CRUD, or row selection is needed. Responsive layout stacks the input before results, permits internal code scrolling, and prevents page-wide overflow. Motion honors reduced-motion.
