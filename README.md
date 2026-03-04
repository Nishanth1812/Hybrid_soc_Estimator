# Hybrid SOC Estimator Dataset Pipeline

PyBaMM pipeline to generate and preprocess SOC estimation datasets for LSTM training.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Generate dataset

Full dataset (75 healthy + 75 degraded):

```bash
python -B -m data_pipeline.build_dataset
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