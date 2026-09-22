# Mendeley Dataset LSTM Training Results

## Summary

This directory contains the trained PyTorch LSTM model for estimating the
state of charge (SOC) of an electric-vehicle lithium-ion battery. The model
was trained remotely on Modal using a Tesla T4 GPU and the prepared Mendeley
EV battery dataset.

The run completed successfully for all 60 requested epochs. The best model
was selected using validation loss and achieved a test-set R² of `0.8711`.
This is a useful research baseline, but the error is still too high for
safety-critical battery-management decisions without additional validation
and improvement.

## Dataset

Source: [Mendeley EV Lithium Ion Battery State of Charge Dataset](../../../datasets/mendeley/raw/EV%20Lithium%20Ion%20Battery%20State%20of%20Charge%20Dataset/charging%20and%20discharging%20%281%29.xlsx)

The workbook was converted into leakage-safe sequence data by splitting at
the battery-cycle level before creating sliding windows. The preprocessing
used:

- Input features: measured voltage, measured current, and ambient temperature
- Ambient temperature converted from °C to K
- SOC converted from percentage to a `[0, 1]` fraction and clipped to bounds
- Sequence length: 100 time steps
- Window stride: 1
- Split: 70% train, 15% validation, 15% test by cycle
- Scaling: input scaler fitted on the training split only

| Split | Samples | Shape |
|---|---:|---|
| Train | 421,338 | `(421338, 100, 3)` |
| Validation | 70,587 | `(70587, 100, 3)` |
| Test | 65,043 | `(65043, 100, 3)` |

Prepared arrays and preprocessing metadata are available in
[datasets/mendeley/processed](../../../datasets/mendeley/processed).

## Model

The trained model is `LSTMSOCEstimator` from
[`models/lstm_soc_model.py`](../../../models/lstm_soc_model.py).

Architecture:

1. LSTM: 3 input features → 64 hidden units
2. Dropout: 0.2
3. LSTM: 64 → 32 hidden units
4. Dropout: 0.2
5. Fully connected layer: 32 → 16 units
6. ReLU activation
7. Fully connected output: 16 → 1 unit
8. Sigmoid output for normalized SOC

The model predicts the SOC at the final time step of each 100-step input
sequence.

## Training configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Platform | Modal |
| GPU | NVIDIA Tesla T4 |
| Requested epochs | 60 |
| Completed epochs | 60 |
| Batch size | 1,024 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Learning-rate reduction | Applied during training |
| Loss | Mean squared error (MSE) |
| Best checkpoint criterion | Lowest validation loss |
| Best epoch | 57 |
| Best validation MSE | 0.007054 |

The remote training entry point is [`modal_train.py`](../../../modal_train.py).

## Test results

The metrics below were calculated on the held-out test split using the best
validation checkpoint. Error values are reported in normalized SOC units;
for example, an MAE of `0.0627` corresponds to approximately 6.27 percentage
points of SOC.

| Metric | Result | Interpretation |
|---|---:|---|
| MAE | 0.0627 | Average error: approximately 6.27 SOC percentage points |
| RMSE | 0.0791 | Larger errors increase the score to approximately 7.91 points |
| R² | 0.8711 | The model explains about 87.1% of test-set SOC variation |
| P95 absolute error | 0.1597 | 95% of errors are at or below approximately 15.97 points |
| Maximum error | 0.2976 | Worst observed error: approximately 29.76 points |
| Mean bias | 0.0098 | Slight average SOC overestimation of approximately 0.98 points |

## Assessment

The training run is healthy: the loss improved substantially, the GPU job
completed without failure, and the checkpoint was downloaded and loaded
successfully. The model is suitable for continued experimentation,
visualization, and comparison with improved architectures or preprocessing.

It should not yet be treated as production-ready for an EV BMS. The average
error is meaningful, but the tail error is large: some predictions can be
almost 30 SOC percentage points away from the target. The next evaluation
should test unseen battery cells and operating conditions, not only unseen
windows from the same dataset distribution.

## Artifacts

- [Best model checkpoint](outputs/best_model.pt)
- [Test metrics](outputs/metrics.json)
- [Training history](outputs/training_history.json)
- [Training log](outputs/training.log)
- [Modal training script](../../../modal_train.py)
