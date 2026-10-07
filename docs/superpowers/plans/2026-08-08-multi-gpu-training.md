# Multi-GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing SOC training pipeline automatically use both Kaggle T4 GPUs when they are available while keeping single-GPU, CPU, evaluation, and deployment behavior compatible.

**Architecture:** Keep the existing `Trainer` and model unchanged. Add small helpers in `training/train_pipeline.py` to wrap the model with `torch.nn.DataParallel` only when CUDA reports more than one GPU, and to unwrap the model before saving checkpoints so `models/best_model.pt` retains the plain `LSTMSOCEstimator` state-dict format expected by evaluation and deployment.

**Tech Stack:** Python 3.11+, PyTorch, `torch.nn.DataParallel`, `unittest`.

## Global Constraints

- Preserve the current LSTM architecture and training hyperparameters.
- Use both CUDA devices automatically when `torch.cuda.device_count() > 1`.
- Fall back to the current single-CUDA-device or CPU behavior when fewer than two GPUs are available.
- Save checkpoints without `module.` prefixes so existing consumers can load them unchanged.
- Do not modify generated datasets, existing model artifacts, or unrelated working-tree changes.

---

### Task 1: Add regression tests for multi-GPU helpers

**Files:**
- Create: `tests/test_multi_gpu_training.py`
- Test: `tests/test_multi_gpu_training.py`

**Interfaces:**
- Consumes: helper functions from `training.train_pipeline`.
- Produces: tests proving multi-GPU detection and checkpoint compatibility.

- [ ] **Step 1: Write the failing tests**

```python
import tempfile
import unittest
from unittest.mock import patch

import torch

from models.lstm_soc_model import LSTMSOCEstimator
from training.train_pipeline import (
    _should_use_data_parallel,
    _state_dict_for_save,
)


class MultiGPUTrainingTests(unittest.TestCase):
    @patch("training.train_pipeline.torch.cuda.device_count", return_value=2)
    @patch("training.train_pipeline.torch.cuda.is_available", return_value=True)
    def test_uses_data_parallel_when_two_cuda_devices_exist(self, _available, _count):
        self.assertTrue(_should_use_data_parallel())

    @patch("training.train_pipeline.torch.cuda.device_count", return_value=1)
    @patch("training.train_pipeline.torch.cuda.is_available", return_value=True)
    def test_does_not_use_data_parallel_with_one_cuda_device(self, _available, _count):
        self.assertFalse(_should_use_data_parallel())

    def test_saved_parallel_state_dict_loads_into_plain_model(self):
        model = torch.nn.DataParallel(LSTMSOCEstimator())
        state_dict = _state_dict_for_save(model)

        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(state_dict, checkpoint.name)
            restored = LSTMSOCEstimator()
            restored.load_state_dict(torch.load(checkpoint.name, weights_only=True))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```text
python -m unittest tests.test_multi_gpu_training -v
```

Expected: FAIL because `_should_use_data_parallel` and `_state_dict_for_save` do not yet exist.

### Task 2: Implement automatic DataParallel training

**Files:**
- Modify: `training/train_pipeline.py`

**Interfaces:**
- Consumes: the existing `LSTMSOCEstimator`, `Trainer`, optimizer, scheduler, and checkpoint path.
- Produces: `_should_use_data_parallel() -> bool`, `_state_dict_for_save(model)`, and a training model that uses all visible CUDA devices when at least two exist.

- [ ] **Step 1: Add the device-selection helper**

```python
def _should_use_data_parallel() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 1
```

- [ ] **Step 2: Add the checkpoint helper**

```python
def _state_dict_for_save(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()
```

- [ ] **Step 3: Wrap the model after moving it to CUDA**

```python
model = LSTMSOCEstimator().to(device)

if _should_use_data_parallel():
    model = nn.DataParallel(model)
    print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
else:
    print(f"Using device: {device}")
```

- [ ] **Step 4: Save the unwrapped state dictionary**

Replace the checkpoint save call with:

```python
torch.save(_state_dict_for_save(model), model_path)
```

- [ ] **Step 5: Run the regression tests**

Run:

```text
python -m unittest tests.test_multi_gpu_training -v
```

Expected: PASS for all tests.

### Task 3: Document Kaggle execution

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the unchanged `run_system.py` command-line interface.
- Produces: clear Kaggle commands showing that `--train-subsample 1` uses all sequences and that two GPUs are selected automatically.

- [ ] **Step 1: Add the multi-GPU Kaggle command**

```text
python -u run_system.py --dataset-path datasets/processed --epochs 30 --batch-size 1024 --train-subsample 1 --val-subsample 1 --test-subsample 1
```

- [ ] **Step 2: Explain the fallback behavior**

Document that the same command uses both visible GPUs through `DataParallel` when two CUDA devices are available, otherwise it falls back to one GPU or CPU.

- [ ] **Step 3: Verify the documented command remains unchanged**

Run the test suite and inspect the final diff to confirm only the intended training and documentation files changed.

### Task 4: Rebuild the uploadable code ZIP

**Files:**
- Create: `Hybrid_soc_Estimator_code.zip`

**Interfaces:**
- Consumes: the updated source tree.
- Produces: a ZIP containing source code, tests, configuration, and documentation, excluding datasets, virtual environments, caches, and checkpoints.

- [ ] **Step 1: Recreate the ZIP from the updated source tree**

Include `tests/` along with the existing source directories and top-level project files.

- [ ] **Step 2: Inspect archive contents**

Confirm that the archive contains `training/train_pipeline.py` and `tests/test_multi_gpu_training.py`, and does not contain `datasets/`, `.venv/`, `__pycache__/`, or `models/best_model.pt`.
