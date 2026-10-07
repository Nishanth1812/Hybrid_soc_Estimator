# 60-second SOC forecast with current SOC input

Trained on the Mendeley workbook on an NVIDIA A10G on 2026-10-02. The
chronological split contains 72,509 train, 96,175 validation, and 94,156 test
windows. Training stopped after epoch 65 and selected epoch 25 by
cycle-balanced validation loss.

| Held-out test measure | LSTM | Hold current SOC constant |
|---|---:|---:|
| Window MAE | 0.0064 pp | 0.3943 pp |
| Macro cycle MAE | 0.0148 pp | 0.7666 pp |
| Worst cycle MAE | 0.0335 pp | 1.2471 pp |

The model forecasts SOC 60 seconds after the final input. Test inputs use the
Mendeley reference SOC at that final timestamp as the current SOC estimate.
Vehicle inference must use a live BMS SOC estimate, so these results do not
include error from an inaccurate BMS estimate. The workbook also represents a
single cell under regulated charge/discharge cycles at constant 24 C, not a
vehicle pack or driving conditions.

Artifacts:

- `best_model.pt` — selected checkpoint and preprocessing metadata
- `forecast_metrics.json` — validation and test metrics by cycle and condition
- `training_history.json` — epoch losses and training configuration
- `training.log` — A10G run log

Modal run: [view app](https://modal.com/apps/nishanthdevabathini1812/main/ap-g0uvd3GFczkR8sObxkhZTK).
