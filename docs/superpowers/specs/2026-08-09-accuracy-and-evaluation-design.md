# Accuracy and Evaluation Design

## Goal

Improve the SOC estimator experiment and make its accuracy measurable without requiring new Kaggle notebook commands.

## Scope

- Increase the default generated dataset to 225 healthy and 225 degraded simulations.
- Increase the default maximum training duration to 60 epochs and early-stopping patience to 12.
- Save training history as JSON.
- Evaluate the saved best checkpoint on the complete test split after training.
- Save MAE, RMSE, and maximum error to JSON.
- Save plots for training history, SOC tracking, prediction parity, and error distribution.

## Design

`run_system.py` remains the entry point. After validation and training, it loads the test split and the best checkpoint, computes predictions, writes `evaluation_outputs/metrics.json`, and saves four PNG files. The current model architecture, DataParallel behavior, plain checkpoint format, simulation split, and preprocessing remain unchanged.

The 3× dataset is generated through the existing configuration defaults. Existing command-line overrides remain respected, so explicit `--num-healthy`/`--num-degraded` values can still be used for smoke tests. The production Kaggle generation command must use the new 225/225 values or omit the overrides to consume the new defaults.

## Success criteria

- The existing validation and training flow completes with the larger dataset.
- A best checkpoint remains loadable by evaluation/deployment.
- Metrics and all four plots are produced automatically after training.
- Test metrics make it possible to decide whether a 4× dataset or model architecture change is warranted.
