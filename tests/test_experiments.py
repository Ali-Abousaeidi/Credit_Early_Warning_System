from __future__ import annotations

import pandas as pd

from src.experiments import is_trend_feature, level_feature_columns, trend_feature_columns


def test_trend_feature_classifier_identifies_window_features() -> None:
    assert is_trend_feature("utilization_slope_6m")
    assert is_trend_feature("cash_buffer_months_change_1m")
    assert not is_trend_feature("credit_score_at_origination")


def test_feature_subset_helpers_partition_columns() -> None:
    columns = [
        "credit_score_at_origination",
        "utilization",
        "utilization_slope_6m",
        "payment_to_due_ratio_std_3m",
    ]

    assert level_feature_columns(columns) == ["credit_score_at_origination", "utilization"]
    assert trend_feature_columns(columns) == [
        "utilization_slope_6m",
        "payment_to_due_ratio_std_3m",
    ]
