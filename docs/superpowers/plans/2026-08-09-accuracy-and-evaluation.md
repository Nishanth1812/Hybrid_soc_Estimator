# Accuracy and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale the default SOC dataset to 450 simulations, train for up to 60 epochs with stronger patience, and automatically produce test metrics and diagnostic plots.

**Architecture:** Keep `run_system.py` as the single entry point. Extend the training pipeline to persist epoch history, add reusable prediction/evaluation helpers, and add plot functions that save files instead of relying on interactive display. After training, the runner evaluates the saved best checkpoint and writes all artifacts under `logs/` and `evaluation_outputs/`.

**Tech Stack:** Python, NumPy, PyTorch, Matplotlib, JSON, `unittest`.

## Global Constraints

- Preserve the existing preprocessing and simulation-level 70/15/15 split.
- Preserve the existing LSTM model and plain checkpoint state-dict format.
- Preserve automatic two-GPU DataParallel fallback behavior.
- Keep evaluation reproducible by using the saved best checkpoint and the full test split.
- Keep Kaggle commands compatible; only the generation count and epoch values need to reflect the new experiment.

---

### Task 1: Add failing tests for evaluation artifacts

**Files:**
- Create: `tests/test_evaluation_outputs.py`

**Interfaces:**
- Consumes: planned functions `save_training_history`, `plot_training_history`, and `plot_prediction_diagnostics`.
- Produces: tests proving history and diagnostic PNG files are written.

- [ ] **Step 1: Write tests**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from evaluation.evaluation_plots import (
    plot_prediction_diagnostics,
    plot_training_history,
)
from training.train_pipeline import save_training_history


class EvaluationOutputTests(unittest.TestCase):
    def test_history_and_diagnostic_plots_are_saved(self):
        history = {
            "epoch": [1, 2],
            "train_loss": [0.10, 0.05],
            "val_loss": [0.12, 0.08],
            "learning_rate": [0.001, 0.001],
            "epoch_seconds": [2.0, 2.1],
        }
        true_soc = np.array([[0.8], [0.7], [0.6]])
        pred_soc = np.array([[0.81], [0.68], [0.59]])

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_training_history(history, output_dir / "history.json")
            plot_training_history(history, output_dir / "training_history.png")
            plot_prediction_diagnostics(true_soc, pred_soc, output_dir)

            expected = [
                "history.json",
                "training_history.png",
                "soc_tracking.png",
                "prediction_scatter.png",
                "prediction_error_histogram.png",
            ]
            for filename in expected:
                self.assertTrue((output_dir / filename).exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail because the new functions do not exist**

```text
python -m unittest tests.test_evaluation_outputs -v
```

### Task 2: Implement history and plotting helpers

**Files:**
- Modify: `evaluation/evaluation_plots.py`
- Modify: `training/train_pipeline.py`

**Interfaces:**
- Produces: `save_training_history(history, path)`, `plot_training_history(history, path)`, and `plot_prediction_diagnostics(true_soc, pred_soc, output_dir)`.

- [ ] **Step 1: Implement JSON history persistence**

Write the history dictionary with `json.dump(..., indent=2)` and create parent directories automatically.

- [ ] **Step 2: Implement four saved plots**

Save training/validation loss, true-vs-predicted SOC tracking, true/predicted parity scatter, and prediction-error histogram using a non-interactive save-and-close workflow.

- [ ] **Step 3: Add history collection to training**

Append epoch number, train loss, validation loss, learning rate, and epoch duration after every epoch. Save `logs/training_history.json` after each epoch and once at completion.

- [ ] **Step 4: Run the focused tests and confirm they pass**

```text
python -m unittest tests.test_evaluation_outputs -v
```

### Task 3: Add reusable prediction and automatic evaluation

**Files:**
- Modify: `evaluation/evaluate_model.py`
- Modify: `run_system.py`
- Create: `tests/test_evaluation_model.py`

**Interfaces:**
- Produces: `predict(model_path, test_loader, device) -> (targets, predictions)` while keeping `evaluate(...) -> dict` compatible.

- [ ] **Step 1: Write a prediction/checkpoint compatibility test**

Create a temporary plain LSTM checkpoint and a tiny dataloader, call `predict`, and assert target/prediction lengths match.

- [ ] **Step 2: Run the test and confirm it fails**

```text
python -m unittest tests.test_evaluation_model -v
```

- [ ] **Step 3: Implement prediction reuse**

Move model loading/inference into `predict`, use `map_location=device`, and make `evaluate` calculate MAE/RMSE/MaxError from `predict` output.

- [ ] **Step 4: Run the focused evaluation tests**

```text
python -m unittest tests.test_evaluation_model tests.test_evaluation_outputs -v
```

- [ ] **Step 5: Wire evaluation into `run_system.py`**

After training, load the full test split, evaluate `models/best_model.pt`, save `evaluation_outputs/metrics.json`, save the four plots, and print their paths and metrics.

### Task 4: Scale defaults and document the experiment

**Files:**
- Modify: `config/config.py`
- Modify: `training/train_pipeline.py`
- Modify: `run_system.py`
- Modify: `README.md`

**Interfaces:**
- Produces: defaults of 225 healthy/225 degraded simulations, 60 maximum epochs, and 12 early-stopping patience while preserving explicit CLI overrides.

- [ ] **Step 1: Update dataset defaults**

Set `SIM_CONFIG["num_healthy"] = 225` and `SIM_CONFIG["num_degraded"] = 225`.

- [ ] **Step 2: Update training defaults**

Set the train pipeline and runner default maximum epochs to 60 and early-stopping patience to 12.

- [ ] **Step 3: Document output paths and the 225/225 command**

Document `logs/training_history.json`, `evaluation_outputs/metrics.json`, and the four plot names. Include the explicit full-data command using 225 healthy and 225 degraded simulations.

- [ ] **Step 4: Run the full test suite and static checks**

```text
python -m unittest discover -s tests -v
git diff --check -- config data_pipeline evaluation run_system.py training tests utils README.md
```

### Task 5: Rebuild the Kaggle upload ZIP

**Files:**
- Rebuild: `Hybrid_soc_Estimator_code.zip`

**Interfaces:**
- Produces: a ZIP containing the updated source/tests/configuration and excluding datasets, caches, logs, and checkpoints.

- [ ] **Step 1: Rebuild the ZIP**

Include `config/`, `data_pipeline/`, `evaluation/`, `models/`, `training/`, `tests/`, `utils/`, top-level scripts, and README.

- [ ] **Step 2: Inspect archive contents**

Confirm evaluation helpers, tests, and updated defaults are present and generated artifacts are absent.
