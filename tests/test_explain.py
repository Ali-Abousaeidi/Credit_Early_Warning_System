from __future__ import annotations

import numpy as np

from src.explain import format_reason, top_positive_contributors


def test_top_positive_contributors_returns_largest_positive_values() -> None:
    contributors = top_positive_contributors(
        np.array([-0.1, 0.5, 0.2]),
        ["a", "b", "c"],
        top_n=2,
    )

    assert contributors == [("b", 0.5), ("c", 0.2)]


def test_format_reason_maps_known_feature_family() -> None:
    reason = format_reason("utilization_slope_6m", 0.08)

    assert "Utilisation trend is rising" in reason
