# Mendeley SOC forecast

The forecaster predicts SOC 60 seconds after the latest observation. It uses a
100-sample history of voltage, signed current, ambient temperature, elapsed
time, and the SOC estimate at the last observed timestamp. When used on a
vehicle, provide the SOC estimate currently maintained by its BMS.

## Prepare the dataset

```bash
python -m pip install -r requirements.txt
python -m soc_estimator.mendeley \
  --input "data/mendeley/raw/EV Lithium Ion Battery State of Charge Dataset/charging and discharging (1).xlsx" \
  --output data/mendeley/forecast_60s_with_current_soc_v1 \
  --train-stride 5
```

The target is reference SOC linearly interpolated at `last input time + 60 s`.
The input SOC is the reference SOC at the last input time; it never includes
future labels. During deployment, provide the live SOC estimate from the BMS.
The model predicts a bounded correction around that estimate. The other four
features are scaled using training cycles only; current SOC is an unscaled SOC
fraction.

Later cycle numbers form chronological, disjoint train, validation and test
splits. Windows cannot cross cycles, duplicate/backward timestamps or gaps over
60 seconds. Targets are omitted where the segment has no reference SOC at the
requested future timestamp. The generated metadata records the split, target
indices, conditions, input scaling and data quality.

## Train on Modal

Upload the generated directory to the existing volume, then start the job in
detached mode:

```bash
modal volume put mahindra-mendeley-bms data/mendeley/forecast_60s_with_current_soc_v1 /
modal run --detach modal_train.py
```

Training runs on an NVIDIA A10G, with a 300-epoch limit, 512-sample batches,
and early stopping after 40 epochs without validation improvement. It selects
and restores the best cycle-balanced validation checkpoint. A new output
folder is used for each run. Validation and test reports compare the LSTM with
both a linear baseline and holding the current SOC estimate constant.

## Inference

Download the new run's `best_model.pt`, then pass a chronological history and
current SOC estimate as a fraction from 0 to 1:

```python
from soc_estimator.evaluation import forecast

# 100 rows: timestamp (s), voltage (V), signed current (A), temperature (C)
result = forecast("best_model.pt", history, current_soc=0.52)
print(result["target_time_s"], result["soc_percent"])
```

The checkpoint includes its model settings and scaler, and inference applies
the same conversions used in training.

## Current evaluation

The current-SOC LSTM was trained on an NVIDIA A10G, stopped after 65 epochs,
and selected epoch 25. On 94,156 chronological held-out windows, its MAE was
0.0064 SOC percentage points (0.0148 points macro-averaged by cycle), compared
with 0.3943 points (0.7666 macro by cycle) for holding the current SOC constant.
The report, checkpoint, history, and training log are in
[`artifacts/mendeley/a10g_forecast_60s_with_current_soc_v1`](artifacts/mendeley/a10g_forecast_60s_with_current_soc_v1/README.md).

Evaluation uses Mendeley's reference SOC at the input timestamp as the current
SOC estimate. A vehicle must supply its live BMS estimate, so its actual forecast
error also depends on that estimate's accuracy. Earlier sensor-only runs are not
comparable to this model because they did not receive current SOC.

This workbook contains regulated cell charge/discharge cycles, not vehicle
packs or driving data. Ambient temperature is constant at 24 C. Test-cycle
scores therefore do not establish vehicle or pack readiness.
