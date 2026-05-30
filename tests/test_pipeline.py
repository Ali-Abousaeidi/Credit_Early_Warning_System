from __future__ import annotations

from src.pipeline import PIPELINE_MODULES


def test_pipeline_order_respects_data_dependencies() -> None:
    assert PIPELINE_MODULES == [
        "src.data_panel",
        "src.eda",
        "src.target",
        "src.features",
        "src.model",
        "src.staging",
        "src.watchlist",
        "src.explain",
        "src.audit",
    ]
