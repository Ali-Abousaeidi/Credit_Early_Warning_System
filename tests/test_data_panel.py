from __future__ import annotations

import pandas as pd

from src.data_panel import PanelConfig, build_synthetic_panel


def test_panel_has_one_row_per_account_month() -> None:
    config = PanelConfig(n_accounts=25, n_months=12, seed=123)
    panel = build_synthetic_panel(config)

    assert len(panel) == config.n_accounts * config.n_months
    assert panel[["account_id", "observation_month"]].duplicated().sum() == 0
    assert panel.groupby("account_id").size().eq(config.n_months).all()


def test_panel_generation_is_reproducible() -> None:
    config = PanelConfig(n_accounts=10, n_months=10, seed=777)

    first = build_synthetic_panel(config)
    second = build_synthetic_panel(config)

    pd.testing.assert_frame_equal(first, second)


def test_panel_excludes_hidden_future_event_dates() -> None:
    panel = build_synthetic_panel(PanelConfig(n_accounts=10, n_months=10, seed=11))

    forbidden_terms = ["future", "target", "label", "event_month", "deterioration_month"]
    lower_columns = [column.lower() for column in panel.columns]

    for term in forbidden_terms:
        assert not any(term in column for column in lower_columns)


def test_status_and_days_past_due_are_consistent() -> None:
    panel = build_synthetic_panel(PanelConfig(n_accounts=20, n_months=14, seed=13))
    expected_dpd = {
        "current": 0,
        "dpd_30": 30,
        "dpd_60": 60,
        "dpd_90": 90,
        "default": 120,
    }

    actual = panel["performance_status"].map(expected_dpd)

    assert actual.notna().all()
    assert (actual == panel["days_past_due"]).all()
