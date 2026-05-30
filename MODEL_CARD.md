# Model Card

## Model

The early-warning model is an XGBoost binary classifier trained to predict whether a currently-performing account will migrate to `30+ DPD` within the next six months.

The model artifact is written to `models/ews_xgb_model.json`.

## Target

- Eligible population: account-month rows where the account is current at observation month `t`.
- Positive event: first migration to `30+ DPD` or worse.
- Horizon: `t+1` through `t+6`.
- Labelled rows: `72,004`.
- Positive rows: `2,558` (`3.55%`).

## Features

The model uses `70` point-in-time features, including:

- Current behavioural levels: utilisation, balance, payment coverage, cash buffer.
- Behavioural trend features: 3- and 6-month slopes, volatility, and changes.
- Missed-minimum-payment counts and flags.
- Prior delinquency recency.
- Static origination context such as original score and origination PD.

Forward labels, future event dates, months-to-deterioration, current DPD metadata, and split labels are excluded from the model feature list.

## Validation

The split is time-based:

- Train: `2020-01-31` to `2021-06-30`
- Calibration: `2021-07-31` to `2021-12-31`
- Test: `2022-01-31` to `2022-06-30`

Calibration uses isotonic regression fitted only on the calibration window.

Out-of-time test metrics:

- ROC-AUC: `0.9597`
- PR-AUC: `0.8107`
- Brier score: `0.0350`
- Log loss: `0.1450`
- Test base rate: `6.21%`

## Watchlist Use

The model is used to rank currently-performing accounts for review. At top `100` accounts per month in the out-of-time test period:

- Monthly precision@100: `92.00%`
- Account capture rate: `78.68%`
- Median lead time: `4.0` months

## IFRS 9 / SICR Framing

The score is mapped to a simplified SICR review trigger using the 95th percentile of calibrated scores in the calibration window. This is a quantitative review signal, not a full IFRS 9 staging policy.

Real IFRS 9 staging is governed, multi-factor, and institution-specific. The model would support that process rather than replace it.

## Limitations

The data is synthetic and intentionally contains visible behavioural drift before deterioration. The model therefore demonstrates a sound early-warning workflow, not real-world production performance.

The highest score decile overpredicts observed deterioration, so calibrated scores should be treated as ranking-oriented risk scores rather than governed probability-of-default estimates.
