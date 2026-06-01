from __future__ import annotations

import pandas as pd

from src.time_to_event import fit_time_to_event_model


def test_time_to_event_model_can_fit_positive_rows() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.4, 0.8],
            "feature_b": [1.0, 0.8, 0.4, 0.2],
            "months_to_deterioration": [6.0, 5.0, 3.0, 1.0],
        }
    )

    model = fit_time_to_event_model(frame, ["feature_a", "feature_b"])
    predictions = model.predict(frame[["feature_a", "feature_b"]].astype(float))

    assert len(predictions) == 4
