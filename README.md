# Credit Early-Warning System

Predicts which currently-performing loans are sliding toward deterioration before they become delinquent, using behavioural trend signals on an account-month panel. The output is a ranked watchlist with lead-time analysis and officer-facing reason codes, framed around IFRS 9 staging / SICR.

## Headline Result

Not yet produced. Metrics in this README will be filled only after running the pipeline end to end.

Target headline:

> Flags deteriorating accounts a median of `[N]` months before they hit 30+ DPD, capturing `[x]%` of true deteriorations at precision@`[k] = [y]%`.

## Phase 0 Scope

This project is an early-warning monitoring model, not an origination scorecard.

- Observation cadence: monthly account snapshots.
- Eligible population: accounts that are currently performing at observation month `t`.
- Warning horizon: the six months after observation, `t+1` through `t+6`.
- Deterioration event: first migration to `30+ days past due` or worse during the horizon.
- Timing wall: all features must use information available up to and including month `t`; labels must use only months after `t`.
- IFRS 9 framing: the model is treated as one data-driven SICR signal that could feed a governed Stage 1 to Stage 2 review. It is not a complete IFRS 9 staging policy.

The default data approach is a reproducible synthetic account-month panel with documented deterioration paths. This avoids fabricating results while keeping the project public, runnable, and point-in-time auditable.

## How It Will Work

1. Account-month monitoring panel.
2. Forward-looking deterioration target with strict point-in-time discipline.
3. Behavioural trend features: utilisation momentum, payment behaviour, volatility, and delinquency onset signals.
4. Calibrated early-warning model evaluated on later observation months.
5. IFRS 9 / SICR mapping.
6. Ranked watchlist, lead-time analysis, and reason codes.

## Synthetic Panel

Phase 1 builds the monitoring panel with:

```bash
python -m src.data_panel
```

Latest generated panel summary:

- Rows: `90,000`
- Accounts: `2,500`
- Observation window: `2020-01-31` to `2022-12-31`
- Months per account: `36`
- Accounts ever reaching `30+ DPD`: `449` (`17.96%`)
- Status rows: `84,689 current`, `596 dpd_30`, `413 dpd_60`, `355 dpd_90`, `3,947 default`

The generated CSV and metadata are written under `data/panel/`, which is git-ignored. Re-run the command above to recreate them from seed `42`.

## Transition Behaviour

Phase 2 calculates one-month state migrations and example deterioration trajectories:

```bash
python -m src.eda
```

Latest generated transition summary:

- One-month transitions analysed: `87,500`
- Current to `30+ DPD` roll rate: `0.79%` per month
- Current to current persistence: `99.21%`
- `30 DPD` cure rate to current: `17.51%`
- `30 DPD` worsening rate to `60+ DPD/default`: `62.00%`

![Transition matrix](reports/figures/transition_matrix.png)

The example trajectory plot shows the intended leading-indicator pattern in the synthetic panel: utilisation tends to rise before the first `30+ DPD` observation, while payment-to-due ratios weaken before delinquency.

![Example deterioration trajectories](reports/figures/example_deterioration_trajectories.png)

## How To Run

The full one-command pipeline will be added as the implementation phases are completed.

```bash
pip install -r requirements.txt
python -m src.data_panel
python -m src.eda
```

## Caveats

- The initial implementation will use synthetic data, clearly labelled as synthetic.
- Synthetic data is useful for demonstrating timing discipline and modelling workflow, but it cannot validate real portfolio performance.
- Real IFRS 9 staging is governed, multi-factor, and policy-specific. This project models one quantitative early-warning signal that would support, not replace, that process.
