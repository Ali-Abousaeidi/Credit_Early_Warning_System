# Interview Notes

## One-Minute Summary

This is an early-warning system for existing credit accounts. It differs from an origination scorecard because it monitors accounts month by month after booking. The model predicts whether a currently-performing account will migrate to `30+ DPD` within the next six months and produces a ranked watchlist for review.

Headline result on synthetic out-of-time data:

- Median lead time: `4.0` months.
- Monthly precision@100: `92.00%`.
- Account capture rate: `78.68%`.

## Timing Wall

For account-month `t`:

- Features use data up to and including `t`.
- Labels use only `t+1` through `t+6`.
- The current month is excluded from the label.
- Rows without a complete six-month future window are excluded from modelling.

This prevents the model from learning outcome-window information.

## Why PR-AUC Matters

Deterioration is a rare event. ROC-AUC can look strong even when the top of the watchlist is not useful. PR-AUC and precision@k are closer to the operational question: if a credit officer reviews the top accounts, how many are true risks?

## Why Lead Time Is the Headline

An EWS is valuable only if it creates time to act. A model that flags borrowers only one month before delinquency may have good discrimination but poor business value. This project reports median months of warning before first `30+ DPD`.

## Why Trend Features Matter

Credit deterioration is a process. A borrower with moderate utilisation that is rising quickly can be riskier than a borrower with high but stable utilisation. The ablation study supports this: trend-only features retain most of the full model's PR-AUC.

## Benchmark Finding

The benchmark layer shows balanced logistic regression performs very strongly on this synthetic dataset. That is not a problem; it means the synthetic deterioration mechanism is partly linear and auditable. XGBoost is retained as the main model artifact because it demonstrates nonlinear modelling, feature interactions, and SHAP reason codes.

## IFRS 9 / SICR Framing

The model is not a full IFRS 9 staging policy. It is a quantitative SICR review signal that could feed a governed process. Real staging would also include policy rules, qualitative overlays, forbearance status, macro scenarios, and governance approvals.

## Caveats To Say Out Loud

- The data is synthetic.
- Absolute model performance is not evidence of real portfolio performance.
- The workflow demonstrates timing discipline, monitoring-panel design, and lead-time evaluation.
- The highest score decile overpredicts observed deterioration, so scores are best treated as risk rankings rather than governed PD estimates.
