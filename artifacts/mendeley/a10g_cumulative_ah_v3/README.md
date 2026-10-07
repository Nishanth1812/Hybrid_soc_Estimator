# A10G run with cumulative charge input

This run used the chronological v3 split and the five inputs recorded in
`data/mendeley/processed_chronological_v3/metadata.json`. Training ran on an
NVIDIA A10G, stopped after 62 epochs, and selected epoch 50 by validation MSE.

The LSTM test MAE was `0.005263` SOC fraction (0.526 percentage points), with
R² `0.9879`. Its largest error was still 37.77 percentage points, concentrated
in later discharge cycles. The LSTM remains a comparison model, not the
recommended estimator for these labels.

The source SOC is reproduced by coulomb counting from the protocol's known
cycle start (0% for charge, 100% for discharge). Capacity was fitted using
training windows only: `1.99981 Ah`. On the untouched chronological test split,
this estimator achieved MAE `0.0001011` SOC fraction (0.0101 percentage
points), RMSE `0.0291` points, 95th-percentile absolute error `0.0063` points,
and maximum error `0.3272` points. The reports are
[`metrics.json`](metrics.json), [`grouped_metrics.json`](grouped_metrics.json),
[`coulomb_counting_metrics.json`](coulomb_counting_metrics.json), and
[`coulomb_counting_grouped_metrics.json`](coulomb_counting_grouped_metrics.json).

This result reflects Mendeley's regulated single-cell charge/discharge
protocol. A vehicle estimate must start from a valid SOC and use the actual
pack capacity; this dataset does not validate arbitrary startup states, pack
configurations, or real driving conditions.
