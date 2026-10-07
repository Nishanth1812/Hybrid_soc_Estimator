# Mendeley Project Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task after the user selects Native execution.

**Goal:** Consolidate active Mendeley SOC source under `soc_estimator/`, organize its data and artifacts, and archive retired files without losing content.

**Architecture:** Keep one flat Python package with modules for Mendeley import, data loading, model, training, and evaluation. Keep the Modal launcher at repository root; group active inputs under `data/`, results under `artifacts/`, and preserved retired files under `archive/legacy/`.

**Tech Stack:** Python 3.11+, PyTorch, Modal, NumPy, pandas, scikit-learn, joblib, openpyxl.

**Spec:** `docs/superpowers/specs/2026-10-01-mendeley-project-layout-design.md`

## Global Constraints

- Keep the workbook import command, Mendeley split and scaling behavior, model architecture, training settings, and evaluation metrics unchanged.
- Keep Modal's remote paths `/data/processed` and `/data/outputs` unchanged.
- Do not delete user data or historical notes.
- Do not change model behavior, dependencies, or the Modal training platform.
- Do not recreate the PyBaMM generator, local runner, deployment, or Kaggle paths removed in the previous cleanup.
- Preserve existing uncommitted user changes; do not commit the shared working tree.
- Do not run tests unless explicitly requested; use syntax parsing, path/reference scans, artifact inventory checks, and `git diff --check` for verification.

## Review Focus

- CLI module path: the documented importer command must resolve to `soc_estimator.mendeley`.
- Model checkpoint compatibility: `LSTMSOCEstimator` state-dict keys and tensor shapes must remain unchanged.
- Modal source packaging: the image must include the full `soc_estimator` package and keep its existing remote volume paths.
- Artifact move completeness: checkpoint, metrics, history, log, and run README must all reach `artifacts/mendeley/`.
- Legacy preservation: moved ignored files must retain their file counts and total byte size under `archive/legacy/`.

---

### Task 1: Consolidate source modules under `soc_estimator`

**Files:**
- Create: `soc_estimator/__init__.py`
- Create: `soc_estimator/data.py` from `training/dataset_loader.py`
- Create: `soc_estimator/evaluation.py` from `evaluation/metrics.py` and `evaluation/evaluate_model.py`
- Create: `soc_estimator/mendeley.py` from `data_pipeline/import_mendeley.py`
- Create: `soc_estimator/model.py` from `models/lstm_soc_model.py` and `config/config_model.py`
- Create: `soc_estimator/training.py` from `training/train_pipeline.py`, `training/trainer.py`, and `utils/logging_utils.py`
- Modify: `modal_train.py`
- Modify: `tests/test_dataset_loader.py`, `tests/test_evaluation_model.py`, `tests/test_import_mendeley.py`, `tests/test_logging.py`, `tests/test_metrics.py`
- Delete after successful moves: old source files and their now-empty package folders

**Interfaces:**
- `soc_estimator.mendeley`: `build_splits(frame, sequence_length=100, stride=1, seed=42)` and CLI `main()`.
- `soc_estimator.data`: `SOCDataset` and `load_dataloaders(dataset_path, batch_size=64, num_workers=2, pin_memory=None)`.
- `soc_estimator.model`: `LSTMSOCEstimator` with the current 3-feature, 64/32-unit LSTM structure.
- `soc_estimator.training`: `Trainer`, `train_model(...)`, `save_training_history(...)`, and `configure_logging(...)`.
- `soc_estimator.evaluation`: `predict(...)`, `evaluate(...)`, and `calculate_metrics(...)`.

- [x] **Step 1: Create the flat package modules** with the stated public names; keep implementation logic and tensor/state-dict shapes unchanged.
- [x] **Step 2: Update internal imports** in the new modules to use `soc_estimator.*` consistently.
- [x] **Step 3: Update `modal_train.py`** to import from the new modules and call `.add_local_python_source("soc_estimator")`.
- [x] **Step 4: Update retained tests' imports** to the package paths; do not add or run tests.
- [x] **Step 5: Remove old module files** only after every active import points to `soc_estimator`.
- [x] **Step 6: Verify package syntax and stale imports.** Parse every active `.py` file with `ast.parse`; search active source for old imports (`data_pipeline`, `evaluation`, `models`, `training`, `utils`, `config`). Expected: no parse errors and no old source imports.

### Task 2: Move active Mendeley data and saved run artifacts

**Files and paths:**
- Move: `datasets/mendeley/` → `data/mendeley/`
- Move: `outputs/mendeley/modal/outputs/best_model.pt` → `artifacts/mendeley/best_model.pt`
- Move: `outputs/mendeley/modal/outputs/metrics.json` → `artifacts/mendeley/metrics.json`
- Move: `outputs/mendeley/modal/outputs/training_history.json` → `artifacts/mendeley/training_history.json`
- Move: `outputs/mendeley/modal/outputs/training.log` → `artifacts/mendeley/training.log`
- Move: `outputs/mendeley/modal/README.md` → `artifacts/mendeley/README.md`
- Modify: `soc_estimator/mendeley.py`, `README.md`, `artifacts/mendeley/README.md`

**Interfaces:**
- Import command: `python -m soc_estimator.mendeley --input <workbook> --output data/mendeley/processed`.
- Modal volume input/output remain `/data/processed` and `/data/outputs`.

- [x] **Step 1: Move the Mendeley input workbook, source archive, processed arrays, scaler, and metadata** to `data/mendeley/` without altering contents.
- [x] **Step 2: Set the import CLI's default output** to `data/mendeley/processed` and update the README command and workbook path.
- [x] **Step 3: Move the saved run files** into `artifacts/mendeley/` and update artifact links and descriptions.
- [x] **Step 4: Verify inventory.** Compare the pre-move and post-move file names, counts, and total byte sizes for the active Mendeley data and artifacts. Expected: all files are present under the new paths.

### Task 3: Archive legacy datasets and run outputs

**Files and paths:**
- Move: `datasets/raw/` → `archive/legacy/datasets/raw/`
- Move: `datasets/processed/` → `archive/legacy/datasets/processed/`
- Move: `datasets/scalers/` → `archive/legacy/datasets/scalers/`
- Move: `evaluation_outputs/`, `graphs/`, `logs/`, `outputs/results/` → corresponding folders under `archive/legacy/`
- Modify: `.gitignore`
- Remove when empty: old `datasets`, `outputs`, and legacy output directories

**Interfaces:**
- No active code reads from the legacy archive.
- `data/` and `archive/legacy/` remain ignored, preserving the current exclusion of local datasets and generated outputs.

- [x] **Step 1: Record the legacy file count and total byte size** for each source folder.
- [x] **Step 2: Move legacy files into `archive/legacy/`** without deleting or rewriting them.
- [x] **Step 3: Update `.gitignore`** to ignore `data/` and `archive/legacy/`, and remove obsolete ignore patterns for paths that no longer exist.
- [x] **Step 4: Verify legacy inventory.** Expected: destination file counts and byte sizes equal the recorded source inventory.

### Task 4: Audit the final layout and instructions

**Files:**
- Modify: `README.md`, `artifacts/mendeley/README.md`, `.gitignore` if needed

- [x] **Step 1: Search active source and current READMEs** for obsolete source and data paths (`data_pipeline`, `datasets/mendeley`, `outputs/mendeley/modal`, `run_system.py`, `build_dataset.py`). Expected: no obsolete active usage paths.
- [x] **Step 2: Parse active Python files** without producing bytecode caches. Expected: no syntax errors.
- [x] **Step 3: Run `git diff --check`.** Expected: no whitespace errors.
- [x] **Step 4: List repository-root directories.** Expected: `soc_estimator`, `tests`, `data`, `artifacts`, `archive`, `docs`, and `.git` only.
- [x] **Step 5: Review `git status --short`** and report the pre-existing unrelated changes separately; leave all work uncommitted.
