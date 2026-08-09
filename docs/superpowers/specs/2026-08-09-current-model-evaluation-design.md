# Current Model Evaluation Design

## Goal

Produce a reproducible evaluation report for the existing LSTM SOC estimator using the current test split and its reference SOC targets.

## Scope

- Keep the existing LSTM architecture, checkpoint format, dataset format, and training flow unchanged.
- Extend the metrics report with MAE, RMSE, R², maximum absolute error, mean bias, error standard deviation, and 95th-percentile absolute error.
- Save diagnostic plots for training history, SOC tracking, true-versus-predicted parity, error distribution, residuals versus reference SOC, and error over test-sample order.
- Continue writing artifacts under `evaluation_outputs/` and keep the existing `metrics.json` and plot names compatible.
- Report only conditions represented by the current processed test data; do not fabricate sensor-disturbance or baseline-comparison results.

## Design

`evaluation.metrics` owns pure NumPy metric functions and returns finite scalar values for flattened reference and prediction arrays. `evaluation.evaluation_plots` owns non-interactive Matplotlib output and keeps plot creation independent from model inference. `run_system.evaluate_trained_model` remains the orchestration boundary: it loads the best checkpoint, computes predictions once, writes the metrics JSON, and saves every plot.

The existing four artifact names remain stable: `training_history.png`, `soc_tracking.png`, `prediction_scatter.png`, and `prediction_error_histogram.png`. Two additional artifacts, `prediction_residuals.png` and `prediction_error_over_samples.png`, expose error behavior over the test set. The parity plot includes the identity line and R² annotation so the result can be placed directly in the evaluation slide.

## Error handling

Metric functions validate that reference and prediction arrays have the same number of finite values. R² returns `0.0` for a constant reference array rather than producing a NaN. Plot helpers reject empty inputs and create their parent directory automatically. The runner continues to fail if the checkpoint or test loader cannot be loaded, because a partial evaluation is not trustworthy.

## Verification

- Unit tests cover all metric values, constant-target R² behavior, and creation of all evaluation artifacts.
- The existing evaluation-model, logging, multi-GPU, and pipeline tests remain unchanged and must pass.
- `python -m unittest discover -s tests -v` and `git diff --check` are run after implementation.
