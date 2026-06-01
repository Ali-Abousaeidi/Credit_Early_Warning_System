from __future__ import annotations

import pandas as pd

from src.calibration import expected_calibration_error


def test_expected_calibration_error_is_zero_for_perfect_bins() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y_true, y_score, n_bins=2) == 0.0
