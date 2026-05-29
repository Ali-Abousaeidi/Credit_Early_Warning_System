"""Forward-looking early-warning target construction.

For each account observed at month ``t``, the target looks only at months
``t+1`` through ``t+horizon``. Features will be built separately from data up
to and including ``t``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WARNING_HORIZON_MONTHS: int = 6
DETERIORATION_DPD_THRESHOLD: int = 30


def load_panel(path: Path) -> pd.DataFrame:
    """Load the account-month panel with parsed dates."""
    return pd.read_csv(
        path,
        parse_dates=["observation_month", "origination_month"],
    )


def add_forward_deterioration_target(
    panel: pd.DataFrame,
    horizon_months: int = WARNING_HORIZON_MONTHS,
    dpd_threshold: int = DETERIORATION_DPD_THRESHOLD,
) -> pd.DataFrame:
    """Attach a forward deterioration target to every account-month row.

    The target equals one if the account reaches ``dpd_threshold`` or worse in
    any future month from ``t+1`` to ``t+horizon``. The observation month ``t``
    is excluded by construction.
    """
    if horizon_months < 1:
        raise ValueError("horizon_months must be at least 1.")

    required = {"account_id", "observation_month", "days_past_due"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    labelled = panel.sort_values(["account_id", "observation_month"]).copy()
    future_deterioration_flags: list[pd.Series] = []
    future_event_month = pd.Series(pd.NaT, index=labelled.index, dtype="datetime64[ns]")

    for step in range(1, horizon_months + 1):
        future_dpd = labelled.groupby("account_id")["days_past_due"].shift(-step)
        future_month = labelled.groupby("account_id")["observation_month"].shift(-step)
        future_deterioration_flags.append(future_dpd.ge(dpd_threshold))
        first_event_mask = future_event_month.isna() & future_dpd.ge(dpd_threshold)
        future_event_month.loc[first_event_mask] = future_month.loc[first_event_mask]
        labelled[f"future_dpd_month_plus_{step}"] = future_dpd

    future_flag_frame = pd.concat(future_deterioration_flags, axis=1)
    labelled[f"target_deterioration_{horizon_months}m"] = (
        future_flag_frame.any(axis=1).astype(int)
    )

    first_offsets = np.full(len(labelled), np.nan)
    for step, flag in enumerate(future_deterioration_flags, start=1):
        needs_first = np.isnan(first_offsets) & flag.to_numpy(dtype=bool)
        first_offsets[needs_first] = step
    labelled["months_to_deterioration"] = first_offsets
    labelled["first_deterioration_month"] = future_event_month

    row_position = labelled.groupby("account_id").cumcount()
    last_position = labelled.groupby("account_id")["account_id"].transform("size") - 1
    labelled["has_full_outcome_window"] = row_position <= (last_position - horizon_months)
    labelled["eligible_at_observation"] = labelled["days_past_due"].eq(0)

    helper_columns = [f"future_dpd_month_plus_{step}" for step in range(1, horizon_months + 1)]
    return labelled.drop(columns=helper_columns)


def build_target_population(
    panel: pd.DataFrame,
    horizon_months: int = WARNING_HORIZON_MONTHS,
    dpd_threshold: int = DETERIORATION_DPD_THRESHOLD,
    require_full_horizon: bool = True,
) -> pd.DataFrame:
    """Return labelled rows eligible for early-warning modelling."""
    labelled = add_forward_deterioration_target(
        panel=panel,
        horizon_months=horizon_months,
        dpd_threshold=dpd_threshold,
    )
    mask = labelled["eligible_at_observation"]
    if require_full_horizon:
        mask &= labelled["has_full_outcome_window"]
    return labelled.loc[mask].reset_index(drop=True)


def summarize_target_population(
    target_population: pd.DataFrame,
    panel: pd.DataFrame,
    horizon_months: int = WARNING_HORIZON_MONTHS,
) -> dict[str, object]:
    """Summarize target prevalence and timing-window exclusions."""
    target_column = f"target_deterioration_{horizon_months}m"
    positive_rows = int(target_population[target_column].sum())
    positive_accounts = int(
        target_population.loc[target_population[target_column].eq(1), "account_id"].nunique()
    )
    full_horizon_cutoff = target_population["observation_month"].max()
    excluded_last_window_rows = int(
        panel.loc[panel["observation_month"].gt(full_horizon_cutoff)].shape[0]
    )
    return {
        "horizon_months": horizon_months,
        "dpd_threshold": DETERIORATION_DPD_THRESHOLD,
        "eligible_rows": int(len(target_population)),
        "positive_rows": positive_rows,
        "positive_rate": round(positive_rows / len(target_population), 4),
        "positive_accounts": positive_accounts,
        "eligible_accounts": int(target_population["account_id"].nunique()),
        "observation_month_min": str(target_population["observation_month"].min().date()),
        "observation_month_max": str(target_population["observation_month"].max().date()),
        "excluded_rows_after_full_horizon_cutoff": excluded_last_window_rows,
        "median_months_to_deterioration_among_positive_rows": round(
            float(
                target_population.loc[
                    target_population[target_column].eq(1), "months_to_deterioration"
                ].median()
            ),
            2,
        ),
    }


def write_target_population(
    target_population: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
    summary: dict[str, object],
) -> None:
    """Write labelled target rows and summary metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    target_population.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Build and write the forward-looking target population."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/panel/account_month_panel.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/targets/target_population.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/target_summary.json"),
    )
    parser.add_argument("--horizon-months", type=int, default=WARNING_HORIZON_MONTHS)
    args = parser.parse_args()

    panel = load_panel(args.panel)
    target_population = build_target_population(
        panel=panel,
        horizon_months=args.horizon_months,
    )
    summary = summarize_target_population(
        target_population=target_population,
        panel=panel,
        horizon_months=args.horizon_months,
    )
    write_target_population(target_population, args.output, args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
