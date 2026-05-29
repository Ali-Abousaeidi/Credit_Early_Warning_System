from __future__ import annotations

import pandas as pd

from src.watchlist import account_level_lead_times, build_monthly_watchlist


def _scored_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["A", "B", "C", "A", "B", "C"],
            "observation_month": pd.to_datetime(
                [
                    "2022-01-31",
                    "2022-01-31",
                    "2022-01-31",
                    "2022-02-28",
                    "2022-02-28",
                    "2022-02-28",
                ]
            ),
            "split": ["test"] * 6,
            "predicted_probability": [0.9, 0.2, 0.7, 0.8, 0.6, 0.1],
            "target_deterioration_6m": [1, 0, 1, 1, 0, 0],
            "months_to_deterioration": [3.0, pd.NA, 2.0, 2.0, pd.NA, pd.NA],
            "first_deterioration_month": pd.to_datetime(
                ["2022-04-30", pd.NaT, "2022-03-31", "2022-04-30", pd.NaT, pd.NaT]
            ),
        }
    )


def test_monthly_watchlist_keeps_top_k_per_month() -> None:
    watchlist = build_monthly_watchlist(_scored_rows(), top_k_per_month=2)

    assert len(watchlist) == 4
    assert watchlist.groupby("observation_month").size().eq(2).all()
    assert watchlist.loc[watchlist["observation_month"].eq(pd.Timestamp("2022-01-31")), "account_id"].tolist() == [
        "A",
        "C",
    ]


def test_lead_time_uses_earliest_flag_for_account() -> None:
    watchlist = build_monthly_watchlist(_scored_rows(), top_k_per_month=2)
    lead_times = account_level_lead_times(watchlist)

    account_a = lead_times.set_index("account_id").loc["A"]

    assert account_a["lead_time_months"] == 3.0
    assert account_a["first_flag_month"] == pd.Timestamp("2022-01-31")
