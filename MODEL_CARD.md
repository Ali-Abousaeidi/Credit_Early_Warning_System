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

Calibration comparison on the out-of-time test split:

- Raw XGBoost Brier: `0.1248`, ECE: `0.2331`
- Isotonic Brier: `0.0350`, ECE: `0.0547`
- Platt Brier: `0.0445`, ECE: `0.0703`

Interpretation: calibration materially improves the raw boosted scores, but even the best calibrated score still overpredicts average event frequency. The score is therefore best treated as a review-ranking signal, not a governed PD.

## Benchmarks and Ablation

Additional experiments compare the production-style tree model against simpler baselines and feature subsets.

Benchmark PR-AUC:

- Balanced logistic regression: `0.8616`
- XGBoost benchmark: `0.8389`
- Random forest: see `reports/tables/benchmark_ablation_results.csv`
- Transparent rule score: see `reports/tables/benchmark_ablation_results.csv`

Ablation PR-AUC:

- Full feature set: `0.8389`
- Level-only features: `0.7802`
- Trend-only features: `0.8314`

Interpretation: trend features carry most of the signal, which supports the project thesis that deterioration is better captured through direction and momentum than static levels alone. The transparent logistic benchmark is strong on this synthetic dataset, so it should be treated as a serious challenger model rather than dismissed.

## Watchlist Use

The model is used to rank currently-performing accounts for review. At top `100` accounts per month in the out-of-time test period:

- Monthly precision@100: `92.00%`
- Account capture rate: `78.68%`
- Median lead time: `4.0` months

Capacity sensitivity is reported in `reports/tables/capacity_sensitivity.csv`.

## Time-to-Event Add-On

An auxiliary positive-case model estimates months to deterioration for accounts already labelled as deteriorating within the six-month horizon. This is not a full censored survival model, but it demonstrates the time-to-event extension:

- Mean absolute error: `0.723` months
- Median absolute error: `0.561` months

## IFRS 9 / SICR Framing

The score is mapped to a simplified SICR review trigger using the 95th percentile of calibrated scores in the calibration window. This is a quantitative review signal, not a full IFRS 9 staging policy.

Real IFRS 9 staging is governed, multi-factor, and institution-specific. The model would support that process rather than replace it.

## Limitations

The data is synthetic and intentionally contains visible behavioural drift before deterioration. The model therefore demonstrates a sound early-warning workflow, not real-world production performance.

The highest score decile overpredicts observed deterioration, so calibrated scores should be treated as ranking-oriented risk scores rather than governed probability-of-default estimates.
