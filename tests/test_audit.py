from __future__ import annotations

import pandas as pd

from src.audit import (
    audit_feature_columns,
    audit_raw_panel,
    audit_time_split,
    audit_watchlist_capacity,
)


def test_raw_panel_audit_fails_on_future_column() -> None:
    panel = pd.DataFrame(
        {
            "account_id": ["A"],
            "observation_month": pd.to_datetime(["2022-01-31"]),
            "future_default_flag": [1],
        }
    )

    result = audit_raw_panel(panel)

    assert result["status"] == "fail"


def test_feature_audit_excludes_target_columns() -> None:
    frame = pd.DataFrame(
        {
            "account_id": ["A"],
            "observation_month": pd.to_datetime(["2022-01-31"]),
            "target_deterioration_6m": [1],
            "months_to_deterioration": [3.0],
            "has_full_outcome_window": [True],
            "eligible_at_observation": [True],
            "utilization_slope_6m": [0.1],
        }
    )

    result = audit_feature_columns(frame)

    assert result["status"] == "pass"


def test_time_split_audit_requires_ordered_windows() -> None:
    scored = pd.DataFrame(
        {
            "split": ["train", "calibration", "test"],
            "observation_month": pd.to_datetime(["2021-01-31", "2021-07-31", "2022-01-31"]),
        }
    )

    result = audit_time_split(scored)

    assert result["status"] == "pass"


def test_watchlist_capacity_audit_detects_over_capacity_month() -> None:
    watchlist = pd.DataFrame(
        {
            "observation_month": pd.to_datetime(["2022-01-31", "2022-01-31"]),
            "account_id": ["A", "B"],
        }
    )

    result = audit_watchlist_capacity(watchlist, top_k_per_month=1)

    assert result["status"] == "fail"
