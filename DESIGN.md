---
version: alpha
name: AI Content Breaker
description: A minimal workbench for local text analysis and pipeline inspection.
colors:
  paper: "#ffffff"
  canvas: "#f6f7f9"
  ink: "#233343"
  muted: "#657180"
  line: "#e2e6eb"
  primary: "#244b68"
  blue-hover: "#183a54"
  blue-wash: "#e9eff4"
  ochre: "#a27624"
  ochre-wash: "#f5eddd"
  green: "#39725b"
  danger: "#a33a34"
  scroll-thumb: "#b8c2cc"
  scroll-track: "#eff2f5"
  scroll-hover: "#7d8d9b"
  scroll-active: "#566d80"
typography:
  display:
    fontFamily: "'Manrope Variable', sans-serif"
    fontSize: "34px"
    fontWeight: 650
    lineHeight: 1.35
    letterSpacing: "-1.15px"
  title:
    fontFamily: "'Manrope Variable', sans-serif"
    fontSize: "13px"
    fontWeight: 700
    letterSpacing: "-0.1px"
  body:
    fontFamily: "'IBM Plex Sans Variable', sans-serif"
    fontSize: "13px"
    lineHeight: 1.5
  label:
    fontFamily: "'IBM Plex Sans Variable', sans-serif"
    fontSize: "11px"
    fontWeight: 500
  button:
    fontFamily: "'IBM Plex Sans Variable', sans-serif"
    fontSize: "13px"
    fontWeight: 500
  mono:
    fontFamily: "'IBM Plex Mono', monospace"
    fontSize: "13px"
    fontWeight: 400
  data:
    fontFamily: "'IBM Plex Mono', monospace"
    fontSize: "12px"
    fontWeight: 400
rounded:
  meter: "3px"
  badge: "4px"
  switch: "5px"
  control: "6px"
  icon: "7px"
  rail: "9px"
  panel: "10px"
spacing:
  action-gap: "8px"
  compact-panel: "18px"
  panel: "23px"
  page: "40px"
  page-compact: "20px"
  page-max: "1490px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.paper}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "11px 14px"
  button-primary-hover:
    backgroundColor: "{colors.blue-hover}"
  button-analyze:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.paper}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "0 15px"
    height: "43px"
    width: "100%"
  button-humanize:
    textColor: "{colors.primary}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "0 15px"
    height: "43px"
    width: "100%"
  button-outline:
    textColor: "{colors.primary}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "11px 14px"
  button-text:
    textColor: "{colors.primary}"
    padding: "4px 0"
  button-clear:
    textColor: "{colors.muted}"
    padding: "4px 0 4px 8px"
  input-passage:
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "18px"
    height: "303px"
    width: "100%"
  nav-link:
    textColor: "{colors.muted}"
    rounded: "{rounded.control}"
    padding: "11px 12px"
  nav-link-active:
    textColor: "{colors.primary}"
  report-badge:
    textColor: "{colors.muted}"
    rounded: "{rounded.badge}"
    padding: "4px 8px"
  panel:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.panel}"
  pipeline-rail:
    rounded: "{rounded.rail}"
    padding: "20px 8px"
  stage-disclosure:
    padding: "18px 23px"
  feature-values:
    typography: "{typography.data}"
  rule-values:
    typography: "{typography.data}"
---

# Design System: AI Content Breaker

## Overview

**Creative North Star: "Analysis workbench"**

The interface presents source text beside an inspectable report. Thin borders, cool neutral surfaces, and restrained blue actions keep the pipeline readable during a course demonstration. The rail shows M1–M6 in processing order and M7 Report, which links to the verdict block. Humanize text is a second action. Its M8 translation block sits above the analysis report and is not a rail item.

This is a product interface for local use on Windows. The current interface and supported analysis text are English; no market-specific visual or business assumptions are established. Copy uses direct labels and instructions: “Text analysis,” “Analyze text,” “Humanize text,” “No analysis yet,” and “M5a rule verdict” with “M5b model verdict,” each Human, AI, Abstain, or Unavailable. The M8 badge reads Passed, Already human, Not passed, or Abstain. Avoid promotional or poetic language.

**Token ownership/runtime mapping:** [frontend/styles.css](frontend/styles.css) is canonical. Its root properties define the named colors and heading/mono families; its component selectors own size, spacing, radius, and state values. This file mirrors those values and documents their use. `colors.primary` maps to CSS `--blue`; all other frontmatter color names match their CSS property names. [frontend/main.tsx](frontend/main.tsx) loads the bundled fonts. [frontend/App.tsx](frontend/App.tsx) owns component semantics and state; [UX-CONTRACT.md](UX-CONTRACT.md) owns the workflow. There is no generated theme adapter. Change CSS first and update this document and `.impeccable/design.json` in the same change; compare frontmatter against the named CSS properties and component selectors before running `designmd lint`.

**Key Characteristics:**

- Clear separation between source text, model readiness, and results.
- A visible M1–M7 rail. M1–M6 follow the pipeline; M7 is the report. M8 is the translation block, not an eighth rail item.
- Flat surfaces, thin borders, and limited color accents.
- Direct labels, explicit states, and inspectable technical output.

## Colors

Cool white and slate surfaces support deep blue actions. Ochre is reserved for focus and caution; green and red communicate status with accompanying text or icons.

### Primary

- **Deep blue** (`primary`, CSS `--blue`): primary action, active navigation, stage codes, and score values. `blue-hover` darkens enabled primary actions; `blue-wash` fills stage labels.

### Secondary

- **Ochre** (`ochre`): keyboard focus, text caret, and the small brand accent. `ochre-wash` supports the stale-report notice.

### Neutral

- **White** (`paper`): source and report panels and foreground text on the primary action.
- **Cool gray** (`canvas`): application background.
- **Slate ink** (`ink`): body text and primary data labels.
- **Muted slate** (`muted`): supporting copy, metadata, and inactive navigation.
- **Light gray** (`line`): panel borders, dividers, and table rules.
- **Scrollbar tones** (`scroll-thumb`, `scroll-track`, `scroll-hover`, `scroll-active`): global document and nested overflow treatment. Standards-based thin scrollbars have an 8px WebKit fallback.

### Semantic status

- **Green** (`green`): ready model, sufficient text, and complete report states.
- **Red** (`danger`): validation and request errors; also the enabled Clear action on hover.

The implementation is light-only. Forced-colors mode restores automatic scrollbar colors, uses the system highlight for filled data marks, and adds a system-colored button border. Selection uses a pale ochre highlight. Individual chart and token-category tints remain component CSS values rather than a second global palette.

**The Status Text Rule.** Pair status colors with a label, icon, or explicit value; a score color must never imply proof of authorship.

## Typography

**Display Font:** Manrope Variable (sans-serif fallback). Used for the brand, page title, panel headings, model state, and report verdict.

**Body Font:** IBM Plex Sans Variable (sans-serif fallback). Used for prose, navigation, fields, controls, and table labels.

**Label/Mono Font:** IBM Plex Mono (monospace fallback). Used for stage identifiers, feature names, numeric values, parse trees, and original console output. The loaded mono weight is regular.

The families distinguish headings, reading, and technical output without exaggerated scale changes. Fonts are bundled through Fontsource; the interface does not depend on a remote font service.

### Hierarchy

- **Display:** `display` records the desktop page title's 34px maximum. CSS uses `clamp(24px, 2.3vw, 34px)`; it changes to 27px below 1060px and 26px below 700px. Mobile line-height is 1.4. The responsive expression stays in prose because the current linter accepts fixed dimensions only.
- **Title:** `title` is the panel heading. The report verdict uses Manrope at 17px, weight 650, line-height 1.6; it becomes 16px on mobile.
- **Body:** `body` describes page guidance. The desktop passage field uses the same family and size with line-height 1.85; mobile input uses 16px and line-height 1.7.
- **Label:** `label` describes model and field labels. Navigation and full-size action text use 13px; contextual metadata is smaller within its existing component.
- **Mono:** `mono` describes default metric values. Summary counts use 17px and score values use 14px. Parse trees and console output use 10px with generous line-height and internal scrolling.

Feature values, feature names, and scoring table values use `data` at every viewport. Table headers use IBM Plex Sans at the same size. Preserve this report-data size independently of compact metadata.

Use sentence case. Actual stage identifiers, code, and domain abbreviations retain their own casing. Numeric formatting currently uses `en-IN`; percentages and decimal precision follow the report formatter rather than a separate visual convention.

## Layout

The desktop shell has a sticky 224px sidebar and a flexible content column. The top bar is 65px tall. Main content has a maximum width of `page-max` with `page` padding; its 40px top padding becomes 50px at 1600px and above. Seven equal rail items sit above a source/report grid with columns `0.86fr 1.14fr` and a 23px gap. Panel headings are 61px tall, and most panel content uses `panel` padding.

The spacing is component-specific; there is no enforced 4px or 8px scale. Shared roles in frontmatter record actual values. Keep document scrolling for the page, with internal scrolling for long tokens, trees, tables, training logs, and console text. The root reserves the scrollbar gutter and the shell uses `minmax(0, …)` to keep wide report content contained.

- **At 1280px and below:** sidebar becomes 200px; main side padding is 25px; panel padding and the source/report gap reduce to 18px. Rail descriptions are hidden.
- **At 1060px and below:** sidebar becomes 180px; the source and report stack in that order; main side padding is 22px; the guide uses three columns.
- **At 700px and below:** sidebar becomes an in-flow header with a compact model panel; only the active sidebar link remains visible. Main side padding is `page-compact`, the top bar becomes 43px, and the guide uses two columns. The rail uses three columns, so all seven items remain visible by wrapping.
- **At 1600px and above:** passage field height grows from 303px to 355px. Its height is 230px at the tablet breakpoint and 255px on mobile.

The empty report has a deliberate minimum height (640px desktop, 430px tablet, 415px mobile; 692px on wide displays). Feedback reserves a 24px form region; the Analyze action remains 43px tall during processing. The complete report grows with its contents.

## Elevation & Depth

There are no box shadows. White panels, pale section fills, and one-pixel borders establish hierarchy. The desktop sidebar is sticky; the page itself owns vertical scrolling. The top bar has a translucent white background without backdrop blur. Keep secondary output visually contained through background tone and borders.

**The Flat Surface Rule.** Use the existing surface fills and borders to separate content; retain the implemented shadow-free treatment.

## Shapes

Panels use `panel` corners, the pipeline rail and model panel use `rail` corners, and buttons, navigation links, and the passage field use `control` corners. Badges use `badge`, report switches use `switch`, and stage icon containers use `icon`. Meters and token containers use the smaller `meter` radius. Status indicators are circles. These roles are distinct; do not replace them with a single global pill shape.

One-pixel strokes define panel edges and section boundaries. Lucide line icons align with text; filled areas are limited to existing uses such as the Play icon and chart marks. The larger empty-report icon container is a component-specific exception rather than a reusable container radius.

## Components

### Foundational visual states

Enabled controls have explicit hover styling. Buttons, links, summaries, the textarea, and focusable console output share a visible ochre focus outline (3px, offset 4px). Enabled buttons move down 1px while pressed. Disabled buttons retain geometry, reduce opacity to 0.5, and use a not-allowed cursor.

Busy states use the app's Lucide spinner, changing text, and disabled controls. The empty-report mark and rail icon fills pulse only during analysis. No skeleton is implemented. Errors remain inline in red; stale results use a pale ochre notice. Success feedback uses green with an icon or text. The report view uses `aria-pressed` and a white bordered selected button.

### Buttons

The primary Analyze action is full width with a 43px fixed height, deep blue fill, white text, a leading Play or loading icon, and a trailing arrow. Its hover color is `blue-hover`. Humanize text uses the same width, height, and trailing arrow, with the outline treatment: a gray-blue border, blue text, a pen icon, and a pale blue hover fill. It sits 8px below Analyze. Its busy label is “Humanizing text…”. The outlined model action has the same border treatment at a shorter height. Load sample is a compact blue text action with an underline on hover. Clear is muted until its enabled red hover state. Copy report and Copy text replace their labels with “Copied”. Use in editor is a text action beside Copy text.

### Chips and status

The compact report badge labels “Awaiting text,” “Processing,” “Complete,” or “Text edited.” Its completed state uses green text and a pale green fill. M1–M7 labels and lexer tokens are data identifiers, not interactive filter chips. Token categories pair their tints with visible type labels.

### Cards / Containers

Source and report panels share white fill, a `line` border, and `panel` corners. Panel headers use an icon and a compact Manrope heading above a bottom divider. Nested parse trees and console output use pale gray fills and internal scrolling. Maintain separate source and result regions when the layout stacks.

### Inputs / Fields

The labeled passage textarea uses a near-white fill, gray-blue border, `control` corners, and fixed responsive height. Hover strengthens the border; focus changes it to blue and adds the common outline. Invalid input uses the danger border and an adjacent inline message. Loading disables the field. Word and character counts remain immediately below it; the 50-word requirement is stated in text and changes to green with a check when met.

### Navigation

Sidebar links use an icon, label, and rounded row. Hover adds a pale fill; the current Text analysis link has a stronger pale blue fill, blue text, and increased weight. Breadcrumbs are subdued and do not compete with the title. The rail represents M1 through M7. Before analysis every item opens the guide. After analysis, M1–M6 open their stage disclosures and M7 scrolls to the verdict block. The mobile transformation is documented in Layout.

### Pipeline disclosures and data

Native `details`/`summary` rows show a stage code, title, concise summary, and chevron. Hover adds a slight gray fill; the chevron rotates with the disclosure state. Stage bodies expose counts, tokens, word frequencies, parse structures, feature values, and scoring rules. Feature lists and rule tables use 12px text on desktop and mobile. Tables use muted headers and horizontal borders; technical content scrolls inside its region.

The verdict block is labeled M7 · Report and states the M5a rule verdict before the M5b model verdict. The translation block is labeled M8 · Translation. A native select chooses Before humanizing report or After humanizing report. The chosen report uses the same verdict heading, score meters, and passage field as the rest of the workbench. The translation-pass table sits below that report, with the same row padding and wrapping behavior as the robustness table. A stale rewrite uses the same pale ochre notice as a stale analysis. Score meters have a thin track with a visible midpoint at the fixed 0.5 threshold. The rule meter uses the ochre fill and the model meter uses the blue-gray fill. Below the word gate, both display an em dash and explicit abstention text. The robustness table uses the same 12px faces as the rule table and wraps inside the stage, with no horizontal scrollbar. Preserve the visible statement that scores are model outputs, not proof of authorship.

### Iconography and motion

Icons come from `lucide-react`; normal inline sizes range from 13px to 19px, with deliberate larger brand and empty-report marks. Icons supplement labels on actions. Color and border changes and disclosure rotation use 150ms ease. The spinner rotates over 1s linear; analysis fill pulses use 1.8s ease-in-out with sequential rail delays. Reduced-motion mode disables animation, transitions, and smooth scrolling.

### Content

Name the operation or state directly. Use the implemented vocabulary consistently: “Check model,” “Check & train model,” “Analyze text,” “Humanize text,” “Pipeline,” “Console,” “Copy report,” “Copy text,” and “Use in editor.” Empty results state “No analysis yet” and explain the next action. Preserve explicit Human, AI, unavailable, and abstention wording, and the M8 statuses Passed, Already human, Not passed, and Abstain. Pipeline terms belong here because inspecting the compiler stages is the user's task. State that a human verdict after rewriting is this detector’s output.

## Do's and Don'ts

### Do:

- **Do** keep the source, results, and model state visually distinct.
- **Do** use the actual M1–M6 stages, the M7 report label, the M8 translation block, and direct action labels.
- **Do** keep visible keyboard focus and explicit status text.
- **Do** keep technical data scrollable within its report region.
- **Do** update this record with the canonical CSS when shared visual values change.

### Don't:

- **Don't** add slogans, poetic empty states, or promotional copy.
- **Don't** describe experimental scores as confidence or proof of authorship.
- **Don't** add shadows to the current flat panel system.
- **Don't** use color as the sole explanation of a verdict or state.
- **Don't** convert component-specific metadata sizes or decorative exceptions into global defaults.
