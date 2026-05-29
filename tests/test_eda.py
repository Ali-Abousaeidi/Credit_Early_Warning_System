from __future__ import annotations

from src.data_panel import PanelConfig, build_synthetic_panel
from src.eda import calculate_transition_counts, calculate_transition_probabilities


def test_transition_counts_have_expected_shape_and_total() -> None:
    config = PanelConfig(n_accounts=30, n_months=12, seed=202)
    panel = build_synthetic_panel(config)

    counts = calculate_transition_counts(panel)

    assert counts.shape == (5, 5)
    assert int(counts.to_numpy().sum()) == config.n_accounts * (config.n_months - 1)


def test_transition_probabilities_are_row_normalized() -> None:
    config = PanelConfig(n_accounts=40, n_months=14, seed=303)
    panel = build_synthetic_panel(config)
    counts = calculate_transition_counts(panel)

    probabilities = calculate_transition_probabilities(counts)
    non_empty_rows = counts.sum(axis=1) > 0

    assert probabilities.loc[non_empty_rows].sum(axis=1).round(8).eq(1.0).all()
