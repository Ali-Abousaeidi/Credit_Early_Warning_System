from __future__ import annotations

import pandas as pd

from src.target import add_forward_deterioration_target, build_target_population


def _toy_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "observation_month": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",
                    "2020-04-30",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",
                    "2020-04-30",
                ]
            ),
            "days_past_due": [0, 0, 30, 60, 0, 0, 0, 0],
            "performance_status": [
                "current",
                "current",
                "dpd_30",
                "dpd_60",
                "current",
                "current",
                "current",
                "current",
            ],
        }
    )


def test_forward_target_excludes_observation_month() -> None:
    labelled = add_forward_deterioration_target(_toy_panel(), horizon_months=1)
    target = labelled.set_index(["account_id", "observation_month"])[
        "target_deterioration_1m"
    ]

    assert target.loc[("A", pd.Timestamp("2020-01-31"))] == 0
    assert target.loc[("A", pd.Timestamp("2020-02-29"))] == 1
    assert target.loc[("A", pd.Timestamp("2020-03-31"))] == 1


def test_target_population_keeps_only_current_rows_with_full_window() -> None:
    population = build_target_population(_toy_panel(), horizon_months=2)

    assert population["days_past_due"].eq(0).all()
    assert population["has_full_outcome_window"].all()
    assert set(population["account_id"]) == {"A", "B"}
    assert len(population) == 4


def test_forward_target_finds_deterioration_inside_horizon_only() -> None:
    population = build_target_population(_toy_panel(), horizon_months=2)
    target = population.set_index(["account_id", "observation_month"])[
        "target_deterioration_2m"
    ]

    assert target.loc[("A", pd.Timestamp("2020-01-31"))] == 1
    assert target.loc[("A", pd.Timestamp("2020-02-29"))] == 1
    assert target.loc[("B", pd.Timestamp("2020-01-31"))] == 0
    assert target.loc[("B", pd.Timestamp("2020-02-29"))] == 0
