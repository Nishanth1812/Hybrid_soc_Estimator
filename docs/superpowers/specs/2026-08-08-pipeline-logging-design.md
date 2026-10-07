# Pipeline Logging Design

## Goal

Make dataset generation and model training visibly trackable in Kaggle without changing the Kaggle notebook commands.

## Design

Both pipelines use Python's standard `logging` module through one shared helper. Each run writes timestamped `INFO` messages to the Kaggle console and to a log file under `logs/`:

- `logs/dataset_generation.log`
- `logs/training.log`

Dataset generation logs the configuration, total work, each completed simulation, retries/failures, raw-save progress, preprocessing stages, output shapes, and total duration. Multiprocessing uses completion-order reporting so a slow simulation cannot hide progress from completed simulations.

Training logs the selected device/GPU count, dataset sizes, batch counts, epoch progress, train/validation loss, learning rate, elapsed time, best-checkpoint saves, early stopping, and total duration. Batch progress is reported periodically rather than once per batch to keep Kaggle output readable.

The logging changes do not alter simulation values, preprocessing, training math, model architecture, checkpoint format, or Kaggle commands.

## Failure behavior

If a simulation future fails, the main process logs the simulation task and traceback before re-raising the error. This keeps failures visible while preserving the current fail-fast behavior. Log directories are created automatically relative to the project working directory.

## Verification

Unit tests will verify that the shared logger writes messages to a requested file and that the training progress path can receive a logger without changing returned loss values. Static checks and the existing multi-GPU tests will also be run.
