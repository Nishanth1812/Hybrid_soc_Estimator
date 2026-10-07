# 60-second SOC forecast training

The implementation is ready and the detached A10G run has started.

- Run: https://modal.com/apps/nishanthdevabathini1812/main/ap-AJpJWw9RtPy6XQZwASQcuw
- Modal volume: `mahindra-mendeley-bms`
- Dataset: `/forecast_60s_v1`
- Output: `/outputs_forecast_60s/20261002T093542029818Z`
- Input: 100 past measurements of voltage, signed current, ambient temperature and elapsed time.
- Target: reference SOC interpolated exactly 60 seconds after the last input timestamp.
- Train / validation / test windows: 72,509 / 96,175 / 94,156.
- Training: up to 300 epochs, batch 512, learning rate 0.0003, patience 40, seed 42.

The best validation checkpoint is saved atomically and includes the scaler and
forecast contract. The run evaluates held-out cycles after training, comparing
the LSTM with a causal linear sensor baseline. The known-current-SOC persistence
comparison is a diagnostic only. Final forecast accuracy is not available yet.

Download the completed checkpoint and report with:

```bash
modal volume get mahindra-mendeley-bms /outputs_forecast_60s/20261002T093542029818Z/best_model.pt artifacts/mendeley/a10g_forecast_60s_v1/best_model.pt
modal volume get mahindra-mendeley-bms /outputs_forecast_60s/20261002T093542029818Z/forecast_metrics.json artifacts/mendeley/a10g_forecast_60s_v1/forecast_metrics.json
```

The initial launch was stopped after it exposed a local/cloud scaler-pickle
version mismatch. This run reads plain numeric scaler statistics from dataset
metadata. Existing outputs are preserved.

Verification: 10 correctness checks pass, covering exact future label alignment,
causal inputs, segment boundaries, chronological holdouts, actual importer/inference
preprocessing consistency, checkpoint loading, data loading and metrics. The
full generated arrays and metadata also passed alignment, shape, finite-value
and disjoint chronological-cycle checks. Compilation and `git diff --check` pass.
