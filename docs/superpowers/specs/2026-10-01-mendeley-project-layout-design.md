# Mendeley Project Layout Design

## Objective

Make the repository easy to scan by grouping the active Mendeley SOC workflow
under one package and placing input data, trained artifacts, and retired
simulation files in clearly named locations. Preserve file contents and model
behavior.

## Current flow

`data_pipeline/import_mendeley.py` prepares cycle-separated arrays.
`modal_train.py` launches remote training and evaluation using the model,
training, evaluation, and logging modules spread across six top-level source
folders. The completed run is stored under `outputs/mendeley/modal/outputs`.
The other dataset and output folders are from the retired synthetic simulation
and local evaluation workflows.

## Proposed structure

```text
soc_estimator/
  __init__.py
  data.py             # PyTorch datasets and loaders
  evaluation.py       # checkpoint prediction and SOC metrics
  mendeley.py         # workbook cleanup, cycle split, scaling, sequences
  model.py            # LSTM and its fixed model settings
  training.py         # trainer, training loop, history, logging
modal_train.py        # Modal entry point
tests/                # tests for the retained workflow
data/mendeley/
  raw/
  processed/
artifacts/mendeley/   # checkpoint, metrics, history, log, run README
archive/legacy/       # preserved synthetic datasets and old local outputs
docs/                 # existing design notes and plans
```

Move the current `config`, `data_pipeline`, `evaluation`, `models`, `training`,
and `utils` code into `soc_estimator`. Merge the old model configuration into
`model.py`, the prediction and metric modules into `evaluation.py`, and the
trainer, orchestration, and logging helper into `training.py`.

Move `datasets/mendeley` to `data/mendeley`. Move the contents of
`outputs/mendeley/modal/outputs` and its run README to `artifacts/mendeley`.
Move the ignored, non-Mendeley simulation datasets and old local run folders
(`evaluation_outputs`, `graphs`, `logs`, and `outputs/results`) under
`archive/legacy`; preserve all files there. Update `.gitignore` so data and
archive files stay excluded from version control, as they are today.

## Behavior and interfaces

- Keep the workbook import command, Mendeley split and scaling behavior, model
  architecture, training settings, and evaluation metrics unchanged.
- Update Python imports, Modal source packaging, and tests to use
  `soc_estimator`.
- Change the local importer default and README paths to `data/mendeley`.
- Keep the Modal volume's remote paths (`/data/processed` and `/data/outputs`)
  unchanged.
- Preserve the existing checkpoint, metrics, training history, log, input
  workbook, processed arrays, historical docs, and legacy generated files.

## Acceptance criteria

- Active Python code is under `soc_estimator/`, except for the root Modal
  launcher; tests remain under `tests/`.
- Root-level folders are limited to the package, tests, data, artifacts,
  archive, docs, and Git metadata.
- No references to retired module paths remain in active code or the current
  README; historical plans remain unchanged.
- The saved Mendeley artifacts and source data exist at their new paths.
- The model/training/evaluation behavior and remote Modal mount paths are
  unchanged.

## Alternatives considered

- Keep the separate code folders under a wrapper package: less module merging,
  but still leaves many folders to navigate.
- Add a `src/` layout: conventional for a reusable published Python library,
  but this repository only needs a local package and Modal source upload, so it
  adds packaging setup without a current benefit.

## Constraints

- Do not delete user data or historical notes.
- Do not change model behavior, dependencies, or the Modal training platform.
- Do not recreate the PyBaMM generator, local runner, deployment, or Kaggle
  paths removed in the previous cleanup.
