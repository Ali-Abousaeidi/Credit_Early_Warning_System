from __future__ import annotations

import pandas as pd

from src.staging import assign_ifrs9_stage_signal, determine_sicr_threshold


def test_sicr_threshold_uses_calibration_split_only() -> None:
    scored = pd.DataFrame(
        {
            "split": ["calibration", "calibration", "test"],
            "predicted_probability": [0.10, 0.20, 0.99],
        }
    )

    threshold = determine_sicr_threshold(scored, calibration_quantile=0.5)

    assert round(threshold, 6) == 0.15


def test_stage_signal_prioritizes_stage_3_over_model_trigger() -> None:
    scored = pd.DataFrame(
        {
            "predicted_probability": [0.01, 0.50, 0.70],
            "days_past_due": [0, 0, 90],
        }
    )

    staged = assign_ifrs9_stage_signal(scored, sicr_threshold=0.40)

    assert staged.loc[0, "ifrs9_stage_signal"] == "Stage 1"
    assert staged.loc[1, "ifrs9_stage_signal"] == "Stage 2"
    assert staged.loc[2, "ifrs9_stage_signal"] == "Stage 3"
