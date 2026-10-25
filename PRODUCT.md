# AI Content Breaker

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React frontend requested by the user. Vite and TypeScript provide local development and builds. The existing Python pipeline remains the analysis and training implementation. `npm run dev` starts both services.

## Users

The primary use is a course demonstration, confirmed by the user. Pipeline details must remain prominent and inspectable.

## Product Purpose

Analyze English text through the compiler pipeline and present the M7 report: both verdicts first, then the M1–M6 stages.

## Operating Context

Local development on Windows, with a Python virtual environment. The frontend and backend run on the same computer. No account or paid API is required.

## Capabilities and Constraints

- Check whether the actual saved model can load. When it cannot, the same action starts training with visible status and errors.
- Accept text and display the M7 verdict block, then the lexer, symbol table, parser and templates, feature vector, rule scoring, trained-model scoring, robustness table, and the verdict-first transcript.
- Preserve the 50-word abstention gate and fixed 0.5 classification threshold.
- Preserve the committed training-only model selection and provenance checks.
- Scores are experimental model outputs, not proof of authorship.
- The user requested frequent commits and live browser verification.

## Brand Commitments

Retain the name AI Content Breaker. The requested presentation is professional and minimal, with deliberate character rather than a generic dashboard.

## Evidence on Hand

README.md, src/pipeline.py, models/training_report.json, committed selection reports, and demo_input.txt provide actual product facts and examples. No invented benchmark or confidence claim belongs in the interface.

## Product Principles

- Make the compiler pipeline legible during a demonstration.
- Keep the entered text visible and intact when a request fails.
- Show model readiness and training honestly.
- Keep the original CLI report available alongside the structured report.
