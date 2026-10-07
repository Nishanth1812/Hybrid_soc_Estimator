# Physics informed A10G run

Run: [Modal app](https://modal.com/apps/nishanthdevabathini1812/main/ap-pcNd4GDtM5M1V5mxANh8U3)

The chronological v4 split contains 36,617 training, 96,974 validation, and
94,956 test windows. Capacity was fit on training cycles only at 1.99983 Ah.
The LSTM starts from the Coulomb-count estimate and learns a correction bounded
to ±0.5 percentage points. Checkpoint selection uses validation loss weighted
equally by cycle. The run stopped after epoch 14 and selected epoch 2.

Test errors are percentage points (SOC fractions multiplied by 100):

| Estimator | MAE | RMSE | P95 absolute error | Maximum error | Macro cycle MAE |
|---|---:|---:|---:|---:|---:|
| Trained residual LSTM | 0.0416 | 0.0446 | 0.0792 | 0.2478 | 0.0624 |
| Coulomb-count baseline | 0.0096 | 0.0291 | 0.0055 | 0.3264 | 0.0160 |

The LSTM reduced the worst error on cycle 548 (0.2436 vs 0.3252 points), while
the direct baseline had lower average error overall and for both charge and
discharge groups. The residual model substantially improves the earlier v3
LSTM, but this workbook's labels closely follow integrated current, so the
physics baseline is the stronger reference for this dataset.

This result assumes charge cycles start at 0% SOC, discharge cycles at 100%,
and the fitted capacity is valid. It comes from one cell at constant ambient
temperature and does not establish vehicle-pack performance or validate
unknown starting SOC, temperature variation, aging, or pack imbalance.
