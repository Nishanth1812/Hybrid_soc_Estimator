# Current Model Evaluation Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing LSTM SOC evaluation flow with complete scalar metrics and diagnostic plots for the current test split.

**Architecture:** Keep model inference and the checkpoint format unchanged. Put pure NumPy calculations in `evaluation/metrics.py`, keep all Matplotlib output in `evaluation/evaluation_plots.py`, and let `run_system.evaluate_trained_model` orchestrate one prediction pass into JSON and PNG artifacts.

**Tech Stack:** Python 3.11, NumPy, PyTorch, Matplotlib, `unittest`.

## Global Constraints

- Evaluate only the existing LSTM checkpoint against the existing reference SOC test targets.
- Preserve the current four artifact names and add only the two new diagnostic plots.
- Do not change the dataset format, training math, model architecture, or checkpoint format.
- Do not fabricate sensor-disturbance or baseline-comparison results when those inputs are absent.
- Keep all generated report files under `evaluation_outputs/`.
- Resolve default paths relative to the project directory so Kaggle runs save artifacts predictably.
- The upload ZIP must exclude datasets, logs, caches, and checkpoints and include the updated source and tests.

---

### Task 1: Add complete scalar metrics

**Files:**
- Modify: `evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: same-shaped NumPy-compatible reference and prediction arrays.
- Produces: `calculate_metrics(y_true, y_pred) -> dict[str, float]` with keys `MAE`, `RMSE`, `R2`, `MaxError`, `MeanBias`, `ErrorStd`, and `P95AbsError`.

- [ ] **Step 1: Write the failing metric tests**

```python
import unittest

import numpy as np

from evaluation.metrics import calculate_metrics, r2


class MetricsTests(unittest.TestCase):
    def test_calculate_metrics_returns_expected_values(self):
        y_true = np.array([[1.0], [2.0], [4.0]])
        y_pred = np.array([[2.0], [2.0], [3.0]])

        actual = calculate_metrics(y_true, y_pred)

        self.assertAlmostEqual(actual["MAE"], 2.0 / 3.0)
        self.assertAlmostEqual(actual["RMSE"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(actual["R2"], 0.7)
        self.assertAlmostEqual(actual["MaxError"], 1.0)
        self.assertAlmostEqual(actual["MeanBias"], 0.0)
        self.assertAlmostEqual(actual["ErrorStd"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(actual["P95AbsError"], 1.0)

    def test_r2_returns_zero_for_constant_reference(self):
        self.assertEqual(r2([0.5, 0.5], [0.5, 0.6]), 0.0)

    def test_metrics_reject_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            calculate_metrics([0.1, 0.2], [0.1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m unittest tests.test_metrics -v`

Expected: FAIL because `calculate_metrics` and `r2` do not exist yet.

- [ ] **Step 3: Implement the metric functions**

Add a shared validator that flattens inputs, requires equal non-empty lengths, and rejects non-finite values. Implement MAE, RMSE, maximum absolute error, signed mean bias, population error standard deviation, and `np.percentile(np.abs(error), 95)`. Implement R² as `1 - SSE/SST`, returning `0.0` when the reference variance is zero. Keep the existing `mae`, `rmse`, and `max_error` names as wrappers for compatibility.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python -m unittest tests.test_metrics -v`

Expected: all three tests PASS.

- [ ] **Step 5: Commit the metrics slice**

```text
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add complete SOC evaluation metrics"
```

### Task 2: Add diagnostic plots

**Files:**
- Modify: `evaluation/evaluation_plots.py`
- Modify: `tests/test_evaluation_outputs.py`

**Interfaces:**
- Consumes: reference and prediction arrays plus an output directory.
- Produces: the existing four PNGs plus `prediction_residuals.png` and `prediction_error_over_samples.png`.

- [ ] **Step 1: Extend the plot test before implementation**

Add the two new filenames to the existing `expected` list and assert each generated file has a non-zero size. The test remains deterministic with the existing three reference/prediction pairs.

- [ ] **Step 2: Run the focused output test and verify it fails**

Run: `python -m unittest tests.test_evaluation_outputs -v`

Expected: FAIL because the two new files are not created yet.

- [ ] **Step 3: Implement the two new plots and enrich the scatter plot**

Keep `matplotlib.use("Agg")`. Add a residual scatter plot with a horizontal zero-error line and an error-over-sample plot with zero and ±MAE guide lines. Add the R² value to the parity scatter title or annotation. Reuse flattened arrays, create the output directory, save at 150 DPI, and close every figure.

- [ ] **Step 4: Run the focused output test and verify it passes**

Run: `python -m unittest tests.test_evaluation_outputs -v`

Expected: PASS with all six PNG files present and non-empty.

- [ ] **Step 5: Commit the plotting slice**

```text
git add evaluation/evaluation_plots.py tests/test_evaluation_outputs.py
git commit -m "feat: add SOC residual and error trend plots"
```

### Task 3: Integrate the report and document outputs

**Files:**
- Modify: `run_system.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `predict(...)`, `calculate_metrics(...)`, and `plot_prediction_diagnostics(...)`.
- Produces: `evaluation_outputs/metrics.json` containing all seven metrics and six PNG diagnostic artifacts after the existing training flow completes.

- [ ] **Step 1: Replace the runner's hand-built metric dictionary**

Import `calculate_metrics` and use it on the single prediction result. Preserve the existing `metrics.json` path, add the new keys, print the complete dictionary, and retain training-history plotting when `logs/training_history.json` exists.

- [ ] **Step 2: Document the complete output set**

Update the README accuracy section with the seven metric names and the six PNG filenames. State that values are computed on the reference SOC test split for the current LSTM only, and that condition-specific sensor-disturbance analysis is unavailable unless such data is added to the dataset.

- [ ] **Step 3: Run the full test suite and static checks**

Run: `python -m unittest discover -s tests -v`

Run: `git diff --check -- evaluation run_system.py tests README.md`

Expected: all discovered tests PASS and `git diff --check` produces no output.

- [ ] **Step 4: Commit the integration slice**

```text
git add run_system.py README.md
git commit -m "feat: integrate complete evaluation report"
```

### Task 4: Make Kaggle reruns self-contained

**Files:**
- Modify: `run_system.py`
- Modify: `training/train_pipeline.py`
- Modify: `validation/dataset_validator.py`
- Modify: `evaluation/evaluate_model.py`
- Modify: `README.md`
- Create: `scripts/build_upload_zip.py`
- Create: `tests/test_dataset_validator_outputs.py`
- Create: `tests/test_evaluation_run.py`
- Create: `tests/test_upload_archive.py`
- Rebuild: `Hybrid_soc_Estimator_code.zip`

**Interfaces:**
- Consumes: project-relative CLI paths and the existing checkpoint/dataset layout.
- Produces: saved validation distributions, `--evaluate-only`, reusable full metrics, and a source-only Kaggle upload archive.

- [ ] **Step 1: Save validation distributions and support project-relative paths**

Make `DatasetValidator.run_full_validation(output_path)` save its distribution figure. Add `--model-path`, `--history-path`, `--log-path`, `--evaluation-output-dir`, and `--evaluate-only` to `run_system.py`, resolving relative values against the script directory. Pass `log_path` through `train_model`.

- [ ] **Step 2: Make the reusable evaluator return all metrics**

Change `evaluation.evaluate_model.evaluate` to call `calculate_metrics`, preserving `predict` and the plain checkpoint format.

- [ ] **Step 3: Add the upload archive builder**

Create `scripts/build_upload_zip.py` that includes source directories, tests, docs, and top-level project files under a `Hybrid_soc_Estimator/` archive root while excluding `datasets/`, `.venv/`, `logs/`, `evaluation_outputs/`, caches, and `models/best_model.pt`.

- [ ] **Step 4: Run focused tests and rebuild the archive**

Run: `uv run python -m unittest tests.test_evaluation_run tests.test_dataset_validator_outputs tests.test_evaluation_model tests.test_upload_archive -v`

Run: `uv run python scripts/build_upload_zip.py`

Inspect the archive with `tar -tf Hybrid_soc_Estimator_code.zip` and confirm the updated evaluator/tests are present and generated artifacts are absent.

- [ ] **Step 5: Commit the Kaggle slice**

```text
git add run_system.py training/train_pipeline.py validation/dataset_validator.py evaluation/evaluate_model.py README.md scripts tests docs/superpowers/plans/2026-08-09-current-model-evaluation-plan.md Hybrid_soc_Estimator_code.zip
git commit -m "feat: make evaluation outputs Kaggle-ready"
```

### Checkpoint: Complete

- [ ] Existing unit tests pass.
- [ ] `metrics.json` includes MAE, RMSE, R², max error, bias, error spread, and P95 absolute error.
- [ ] Six diagnostic PNGs are generated by the evaluation path.
- [ ] Validation distributions and all report artifacts are saved under project-relative paths.
- [ ] The upload ZIP contains current source/tests and no generated artifacts.
- [ ] Existing training, model, dataset, and checkpoint behavior is unchanged.
