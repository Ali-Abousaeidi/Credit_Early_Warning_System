from __future__ import annotations

import pandas as pd

from src.features import build_feature_matrix, build_point_in_time_features, feature_columns


def _toy_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-31", periods=4, freq="M")
    for idx, (observation_month, util) in enumerate(
        zip(dates, [0.10, 0.20, 0.40, 0.80], strict=True), start=1
    ):
        rows.append(
            {
                "account_id": "A",
                "observation_month": observation_month,
                "month_index": idx - 1,
                "account_age_months": idx,
                "origination_month": pd.Timestamp("2019-12-31"),
                "credit_score_at_origination": 700,
                "origination_pd": 0.05,
                "monthly_income": 5_000.0,
                "debt_to_income": 0.25,
                "credit_limit": 10_000.0,
                "apr": 0.12,
                "balance": util * 10_000.0,
                "utilization": util,
                "minimum_payment_due": 50.0,
                "scheduled_payment_due": 100.0,
                "actual_payment": 100.0,
                "payment_to_due_ratio": 1.0,
                "purchase_amount": 500.0,
                "cash_buffer_months": 3.0,
                "missed_min_payment": idx == 3,
                "days_past_due": 0,
                "performance_status": "current",
                "macro_stress_index": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_rolling_features_use_current_and_past_only() -> None:
    features = build_point_in_time_features(_toy_panel())

    row_2 = features.loc[features["month_index"].eq(1)].iloc[0]
    row_3 = features.loc[features["month_index"].eq(2)].iloc[0]

    assert round(row_2["utilization_mean_3m"], 6) == 0.15
    assert round(row_3["utilization_mean_3m"], 6) == round((0.10 + 0.20 + 0.40) / 3, 6)
    assert row_2["utilization_mean_3m"] != row_3["utilization_mean_3m"]


def test_recent_first_missed_payment_flag_is_point_in_time() -> None:
    features = build_point_in_time_features(_toy_panel())

    flags = features["first_missed_min_payment_3m"].tolist()

    assert flags == [0, 0, 1, 0]


def test_feature_matrix_preserves_target_without_using_future_columns() -> None:
    panel = _toy_panel()
    target_population = panel[["account_id", "observation_month", "origination_month"]].copy()
    target_population["target_deterioration_6m"] = [0, 1, 0, 0]
    target_population["months_to_deterioration"] = [pd.NA, 2, pd.NA, pd.NA]
    target_population["first_deterioration_month"] = pd.NaT
    target_population["has_full_outcome_window"] = True
    target_population["eligible_at_observation"] = True

    feature_matrix = build_feature_matrix(panel, target_population)

    assert "target_deterioration_6m" in feature_matrix.columns
    assert not any(column.startswith("future_") for column in feature_matrix.columns)
    assert len(feature_matrix) == len(target_population)


def test_feature_columns_exclude_target_and_timing_metadata() -> None:
    panel = _toy_panel()
    target_population = panel[["account_id", "observation_month", "origination_month"]].copy()
    target_population["target_deterioration_6m"] = [0, 1, 0, 0]
    target_population["months_to_deterioration"] = [pd.NA, 2, pd.NA, pd.NA]
    target_population["first_deterioration_month"] = pd.NaT
    target_population["has_full_outcome_window"] = True
    target_population["eligible_at_observation"] = True

    columns = feature_columns(build_feature_matrix(panel, target_population))

    assert "target_deterioration_6m" not in columns
    assert "months_to_deterioration" not in columns
    assert "first_deterioration_month" not in columns
    assert "month_index" not in columns
    assert "days_past_due" not in columns
