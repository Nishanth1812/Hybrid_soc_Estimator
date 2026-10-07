# Pre-SoH Presentation Revision Design

## Goal

Improve the narrative and evidence in `EV_BMS.pptx` while preserving its existing visual system. The revision scope ends immediately before the slide titled `IoT-based SoH Estimation`.

## Scope

- Edit slides 1–8: title framing, outline, problem statement, objective, hybrid rationale, AI-based SoC path, hybrid SoC architecture, and current-model evaluation.
- Preserve slides 9–12 unchanged, including the full SoH section, future work, conclusion, and references.
- Preserve the imported deck’s master/layout hierarchy, background treatment, typography, footer chrome, page markers, and existing shape positions wherever possible.

## Narrative

1. Establish the estimation problem: battery states are inferred from measured signals.
2. State the objective as a measurable, bounded estimator for EV BMS use.
3. Explain the hybrid mechanism: reference estimate + AI candidate + constrained correction.
4. Show the AI-based SoC pipeline using voltage, current, temperature, and recent history.
5. Show how estimator disagreement is converted into a bounded SoC update.
6. Replace the evaluation placeholders with the current model’s held-out test evidence and metrics from `evaluation_outputs/`.

## Current-model evidence

Use the existing local evaluation outputs without inventing a hybrid comparison that has not been measured:

- MAE: 0.1200
- RMSE: 0.1819
- R²: 0.7591
- Plot: `evaluation_outputs/soc_tracking.png`

The evaluation slide will label the result as current-model / held-out-test evidence and retain the existing comparison language only where it does not imply an unmeasured hybrid result.

## Acceptance criteria

- Slides 9–12 remain structurally and visually unchanged.
- Slides 1–8 render without overflow, clipping, or accidental overlap.
- The evaluation slide contains an embedded, legible current-model plot and populated metrics.
- The output is an editable PPTX, not a flattened image deck.
