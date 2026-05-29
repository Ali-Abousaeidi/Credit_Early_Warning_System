"""Point-in-time behavioural feature engineering.

Trend features are computed from each account's history up to and including the
observation month ``t``. The module never uses future target columns to create
features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.target import WARNING_HORIZON_MONTHS

IDENTIFIER_COLUMNS = ["account_id", "observation_month"]
TARGET_COLUMNS = [
    f"target_deterioration_{WARNING_HORIZON_MONTHS}m",
    "months_to_deterioration",
    "first_deterioration_month",
    "has_full_outcome_window",
    "eligible_at_observation",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and return zero where the denominator is zero."""
    ratio = numerator.div(denominator.replace(0, np.nan))
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_slope(values: np.ndarray) -> float:
    """Return the linear slope across an ordered rolling window."""
    mask = ~np.isnan(values)
    valid = values[mask]
    if len(valid) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)[mask]
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x_centered * (valid - valid.mean())) / denominator)


def _rolling_by_account(
    frame: pd.DataFrame,
    column: str,
    window: int,
    aggregation: str,
    min_periods: int = 1,
) -> pd.Series:
    """Apply a rolling aggregation independently within each account."""
    grouped = frame.groupby("account_id", group_keys=False)[column]
    rolling = grouped.rolling(window=window, min_periods=min_periods)
    if aggregation == "mean":
        result = rolling.mean()
    elif aggregation == "std":
        result = rolling.std()
    elif aggregation == "sum":
        result = rolling.sum()
    elif aggregation == "slope":
        result = rolling.apply(_rolling_slope, raw=True)
    else:
        raise ValueError(f"Unsupported rolling aggregation: {aggregation}")
    return result.reset_index(level=0, drop=True)


def _months_since_last_event_by_account(
    frame: pd.DataFrame, event_column: str = "is_30_plus_dpd"
) -> pd.Series:
    """Months since the last prior event, excluding the current month."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, account_frame in frame.groupby("account_id", sort=False):
        event_month_index = account_frame["month_index"].where(
            account_frame[event_column].astype(bool)
        )
        last_prior_event = event_month_index.ffill().shift(1)
        result.loc[account_frame.index] = account_frame["month_index"] - last_prior_event
    return result


def build_point_in_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time behavioural features for every account-month row."""
    required = {
        "account_id",
        "observation_month",
        "month_index",
        "monthly_income",
        "credit_limit",
        "balance",
        "utilization",
        "minimum_payment_due",
        "scheduled_payment_due",
        "actual_payment",
        "payment_to_due_ratio",
        "purchase_amount",
        "cash_buffer_months",
        "missed_min_payment",
        "days_past_due",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    features = panel.sort_values(IDENTIFIER_COLUMNS).copy()
    features["is_30_plus_dpd"] = features["days_past_due"].ge(30).astype(int)
    features["missed_min_payment_int"] = features["missed_min_payment"].astype(int)

    features["balance_to_income"] = _safe_ratio(
        features["balance"], features["monthly_income"]
    )
    features["payment_to_balance"] = _safe_ratio(
        features["actual_payment"], features["balance"]
    )
    features["purchase_to_limit"] = _safe_ratio(
        features["purchase_amount"], features["credit_limit"]
    )
    features["minimum_due_to_balance"] = _safe_ratio(
        features["minimum_payment_due"], features["balance"]
    )
    features["log_monthly_income"] = np.log(features["monthly_income"].clip(lower=1.0))
    features["log_credit_limit"] = np.log(features["credit_limit"].clip(lower=1.0))

    base_series = [
        "utilization",
        "payment_to_due_ratio",
        "cash_buffer_months",
        "purchase_to_limit",
        "balance_to_income",
    ]
    for column in base_series:
        features[f"{column}_change_1m"] = (
            features.groupby("account_id")[column].diff().fillna(0.0)
        )
        for window in (3, 6):
            features[f"{column}_mean_{window}m"] = _rolling_by_account(
                features, column, window, "mean"
            )
            features[f"{column}_std_{window}m"] = _rolling_by_account(
                features, column, window, "std"
            ).fillna(0.0)
            features[f"{column}_slope_{window}m"] = _rolling_by_account(
                features, column, window, "slope", min_periods=2
            ).fillna(0.0)

    for window in (3, 6):
        features[f"missed_min_payment_count_{window}m"] = _rolling_by_account(
            features, "missed_min_payment_int", window, "sum"
        )
        features[f"missed_min_payment_rate_{window}m"] = (
            features[f"missed_min_payment_count_{window}m"] / window
        )
        features[f"any_missed_min_payment_{window}m"] = (
            features[f"missed_min_payment_count_{window}m"].gt(0).astype(int)
        )
        prior_missed_count = (
            features.groupby("account_id")["missed_min_payment_int"]
            .transform(lambda series: series.shift(1).rolling(window, min_periods=1).sum())
            .fillna(0.0)
        )
        features[f"first_missed_min_payment_{window}m"] = (
            features["missed_min_payment_int"].eq(1) & prior_missed_count.eq(0)
        ).astype(int)

    features["prior_30_plus_count_12m"] = (
        features.groupby("account_id")["is_30_plus_dpd"]
        .transform(lambda series: series.shift(1).rolling(12, min_periods=1).sum())
        .fillna(0.0)
    )
    features["ever_30_plus_before_t"] = features["prior_30_plus_count_12m"].gt(0).astype(int)
    features["months_since_last_30_plus"] = _months_since_last_event_by_account(features)
    features["months_since_last_30_plus"] = features["months_since_last_30_plus"].fillna(99.0)

    return features


def feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding identifiers and target metadata."""
    excluded = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS)
    excluded.update(
        {
            "origination_month",
            "performance_status",
            "first_deterioration_month",
            "month_index",
            "days_past_due",
            "is_30_plus_dpd",
            "split",
        }
    )
    return [
        column
        for column in feature_frame.columns
        if column not in excluded and not column.startswith("target_")
    ]


def build_feature_matrix(panel: pd.DataFrame, target_population: pd.DataFrame) -> pd.DataFrame:
    """Join point-in-time features to the eligible labelled target population."""
    features = build_point_in_time_features(panel)
    target_keys = target_population[IDENTIFIER_COLUMNS + TARGET_COLUMNS].copy()
    merged = target_keys.merge(features, on=IDENTIFIER_COLUMNS, how="left", validate="one_to_one")
    if merged.isna().any().any():
        missing_columns = merged.columns[merged.isna().any()].tolist()
        allowed_missing = {"months_to_deterioration", "first_deterioration_month"}
        unexpected_missing = sorted(set(missing_columns).difference(allowed_missing))
        if unexpected_missing:
            raise ValueError(f"Unexpected missing feature values: {unexpected_missing}")
    return merged


def summarize_feature_matrix(feature_matrix: pd.DataFrame) -> dict[str, object]:
    """Summarize the generated feature matrix."""
    columns = feature_columns(feature_matrix)
    trend_columns = [
        column
        for column in columns
        if (
            column.endswith(("_slope_3m", "_slope_6m"))
            or column.endswith(("_change_1m", "_std_3m", "_std_6m"))
            or column.startswith(
                (
                    "missed_min_payment_count_",
                    "missed_min_payment_rate_",
                    "first_missed_min_payment_",
                )
            )
            or column in {"prior_30_plus_count_12m", "months_since_last_30_plus"}
        )
    ]
    target_column = f"target_deterioration_{WARNING_HORIZON_MONTHS}m"
    return {
        "n_rows": int(len(feature_matrix)),
        "n_accounts": int(feature_matrix["account_id"].nunique()),
        "n_feature_columns": int(len(columns)),
        "n_trend_feature_columns": int(len(trend_columns)),
        "positive_rows": int(feature_matrix[target_column].sum()),
        "positive_rate": round(float(feature_matrix[target_column].mean()), 4),
        "observation_month_min": str(feature_matrix["observation_month"].min().date()),
        "observation_month_max": str(feature_matrix["observation_month"].max().date()),
        "sample_trend_features": trend_columns[:12],
    }


def write_feature_matrix(
    feature_matrix: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
    summary: dict[str, object],
    feature_list_path: Path,
) -> None:
    """Write feature matrix, summary, and feature-column list."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    feature_list_path.write_text(
        "\n".join(feature_columns(feature_matrix)) + "\n", encoding="utf-8"
    )


def load_inputs(panel_path: Path, target_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load panel and target population inputs."""
    parse_dates = ["observation_month", "origination_month"]
    panel = pd.read_csv(panel_path, parse_dates=parse_dates)
    target_population = pd.read_csv(
        target_path,
        parse_dates=["observation_month", "first_deterioration_month", "origination_month"],
    )
    return panel, target_population


def main() -> None:
    """Build and write the point-in-time feature matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/panel/account_month_panel.csv"))
    parser.add_argument("--target", type=Path, default=Path("data/targets/target_population.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/feature_matrix.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/feature_summary.json"),
    )
    parser.add_argument(
        "--feature-list",
        type=Path,
        default=Path("reports/tables/feature_columns.txt"),
    )
    args = parser.parse_args()

    panel, target_population = load_inputs(args.panel, args.target)
    feature_matrix = build_feature_matrix(panel, target_population)
    summary = summarize_feature_matrix(feature_matrix)
    write_feature_matrix(
        feature_matrix=feature_matrix,
        output_path=args.output,
        summary_path=args.summary,
        summary=summary,
        feature_list_path=args.feature_list,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
