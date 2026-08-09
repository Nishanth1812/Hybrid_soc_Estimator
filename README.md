# Hybrid SOC Estimator Dataset Pipeline

PyBaMM pipeline to generate and preprocess SOC estimation datasets for LSTM training.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Generate dataset

Full dataset (225 healthy + 225 degraded):

```bash
python -B -m data_pipeline.build_dataset --num-healthy 225 --num-degraded 225 --workers 2
```

Smoke test:

```bash
python -B -m data_pipeline.build_dataset --num-healthy 2 --num-degraded 2
```

Skip raw per-simulation files:

```bash
python -B -m data_pipeline.build_dataset --skip-raw-save
```

## Outputs

- `datasets/raw/*.npz`
- `datasets/raw/metadata.csv`
- `datasets/processed/X_train.npy`
- `datasets/processed/y_train.npy`
- `datasets/processed/X_val.npy`
- `datasets/processed/y_val.npy`
- `datasets/processed/X_test.npy`
- `datasets/processed/y_test.npy`
- `datasets/scalers/input_scaler.pkl`

## Dataset format

- Split by simulation: `70/15/15` (`train/val/test`)
- Input features: `[Voltage, Current, Temperature]`
- Sequence shape: `(num_samples, 100, 3)`
- Target: SOC at the final timestep



# Things to do

-> Change to bilistm after training it once and checking the results

## Kaggle multi-GPU training

The training pipeline automatically wraps the LSTM with `torch.nn.DataParallel`
when two or more CUDA devices are visible. The same code falls back to one GPU
or CPU when fewer devices are available. Checkpoint files are saved in the
plain `LSTMSOCEstimator` format so evaluation and deployment can load them
without a `module.` prefix.

After generating the full dataset, run from the project directory:

```bash
python -u run_system.py --dataset-path datasets/processed --epochs 60 --batch-size 1024 --train-subsample 1 --val-subsample 1 --test-subsample 1
```

The training log prints `Using DataParallel across 2 GPUs` when both Kaggle
T4 GPUs are being used.

## Logs

The existing Kaggle commands automatically write timestamped messages to the
console and to these files:

- `logs/dataset_generation.log` — simulation progress, retries, raw-file saves,
  preprocessing stages, output shapes, and total duration.
- `logs/training.log` — device/GPU information, dataset sizes, batch progress,
  epoch losses, learning rate, checkpoint saves, early stopping, and total
  duration.

## Accuracy outputs

The full training command evaluates the best current LSTM checkpoint
automatically against the reference SOC values in the test split. It writes
these files under `evaluation_outputs/`:

- `metrics.json` — MAE, RMSE, R², maximum absolute error, mean signed bias,
  error standard deviation, and 95th-percentile absolute error.
- `dataset_distributions.png` — voltage, current, temperature, and reference
  SOC distributions used by validation.
- `training_history.png` — training versus validation loss.
- `soc_tracking.png` — reference and predicted SOC over test samples.
- `prediction_scatter.png` — reference SOC versus predicted SOC with R².
- `prediction_error_histogram.png` — prediction error distribution.
- `prediction_residuals.png` — residuals versus reference SOC.
- `prediction_error_over_samples.png` — error trend over test-sample order.

The project creates `logs/`, `models/`, and `evaluation_outputs/` as needed.
All paths are resolved relative to the project directory, so the same command
works after uploading and extracting the code ZIP in Kaggle.

## Kaggle upload and rerun

Upload `Hybrid_soc_Estimator_code.zip`, extract it, and run from the extracted
project directory:

```bash
python -m pip install -r requirements.txt
python -B -m data_pipeline.build_dataset --num-healthy 225 --num-degraded 225 --workers 2
python -u run_system.py --dataset-path datasets/processed --epochs 60 --batch-size 1024 --train-subsample 1 --val-subsample 1 --test-subsample 1
```

If the dataset and checkpoint already exist, regenerate only the saved report
without retraining:

```bash
python -u run_system.py --evaluate-only --dataset-path datasets/processed --model-path models/best_model.pt
```

The generated dataset, checkpoint, logs, metrics, and plots remain in the
project directory and can be downloaded from the Kaggle working files.

The default full dataset is 225 healthy plus 225 degraded simulations. To
generate it explicitly, run:

```bash
python -B -m data_pipeline.build_dataset --num-healthy 225 --num-degraded 225 --workers 2
```
