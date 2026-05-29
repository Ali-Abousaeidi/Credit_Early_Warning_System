from __future__ import annotations

import pandas as pd

from src.model import ModelConfig, add_time_split


def test_time_split_is_strictly_ordered() -> None:
    frame = pd.DataFrame(
        {
            "account_id": ["A", "A", "A", "A"],
            "observation_month": pd.to_datetime(
                ["2021-01-31", "2021-06-30", "2021-09-30", "2022-01-31"]
            ),
            "target_deterioration_6m": [0, 1, 0, 1],
        }
    )
    config = ModelConfig(train_end="2021-06-30", calibration_end="2021-12-31")

    split = add_time_split(frame, config)

    assert split.loc[0, "split"] == "train"
    assert split.loc[1, "split"] == "train"
    assert split.loc[2, "split"] == "calibration"
    assert split.loc[3, "split"] == "test"
    assert split.loc[split["split"].eq("train"), "observation_month"].max() <= pd.Timestamp(
        config.train_end
    )
    assert split.loc[split["split"].eq("test"), "observation_month"].min() > pd.Timestamp(
        config.calibration_end
    )
