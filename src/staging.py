"""IFRS 9 staging and SICR framing helpers.

This module provides a simplified mapping from model output to an IFRS 9 style
review signal. It is not a full IFRS 9 policy: real staging is governed,
multi-factor, and institution-specific.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.model import TARGET_COLUMN


def load_scored_rows(path: Path) -> pd.DataFrame:
    """Load scored model rows with parsed dates."""
    return pd.read_csv(
        path,
        parse_dates=[
            "observation_month",
            "origination_month",
            "first_deterioration_month",
        ],
    )


def determine_sicr_threshold(
    scored: pd.DataFrame,
    calibration_quantile: float = 0.90,
    split_name: str = "calibration",
) -> float:
    """Set the model SICR trigger from calibration-window score distribution."""
    if not 0.0 < calibration_quantile < 1.0:
        raise ValueError("calibration_quantile must be between 0 and 1.")
    calibration_scores = scored.loc[
        scored["split"].eq(split_name), "predicted_probability"
    ]
    if calibration_scores.empty:
        raise ValueError(f"No rows found for split: {split_name}")
    return float(calibration_scores.quantile(calibration_quantile))


def assign_ifrs9_stage_signal(
    scored: pd.DataFrame,
    sicr_threshold: float,
    stage_3_dpd_threshold: int = 90,
    stage_2_dpd_threshold: int = 30,
) -> pd.DataFrame:
    """Assign simplified IFRS 9 stage signal from DPD and model score.

    Stage 3: credit-impaired proxy, ``days_past_due >= 90``.
    Stage 2: delinquency proxy, ``days_past_due >= 30``, or model SICR trigger.
    Stage 1: no delinquency proxy and no model SICR trigger.
    """
    staged = scored.copy()
    staged["sicr_model_trigger"] = staged["predicted_probability"].ge(sicr_threshold)
    staged["ifrs9_stage_signal"] = "Stage 1"
    staged.loc[
        staged["days_past_due"].ge(stage_2_dpd_threshold) | staged["sicr_model_trigger"],
        "ifrs9_stage_signal",
    ] = "Stage 2"
    staged.loc[
        staged["days_past_due"].ge(stage_3_dpd_threshold),
        "ifrs9_stage_signal",
    ] = "Stage 3"
    return staged


def summarize_staging(
    staged: pd.DataFrame,
    sicr_threshold: float,
    calibration_quantile: float,
) -> dict[str, object]:
    """Summarize stage signal counts and out-of-time target capture."""
    test = staged.loc[staged["split"].eq("test")].copy()
    stage_counts = test["ifrs9_stage_signal"].value_counts().sort_index()
    triggered = test["sicr_model_trigger"]
    positives = test[TARGET_COLUMN].astype(int).eq(1)
    captured = int((triggered & positives).sum())
    trigger_count = int(triggered.sum())
    positive_count = int(positives.sum())
    return {
        "sicr_threshold_source": "calibration predicted_probability quantile",
        "calibration_quantile": calibration_quantile,
        "sicr_threshold": round(float(sicr_threshold), 6),
        "test_rows": int(len(test)),
        "test_stage_signal_counts": {
            key: int(value) for key, value in stage_counts.items()
        },
        "test_sicr_trigger_count": trigger_count,
        "test_sicr_trigger_rate": round(float(triggered.mean()), 4),
        "test_positive_rows": positive_count,
        "test_positive_capture_rate": round(captured / positive_count, 4)
        if positive_count
        else 0.0,
        "test_sicr_precision": round(captured / trigger_count, 4)
        if trigger_count
        else 0.0,
        "framing": (
            "The model trigger is a quantitative SICR review signal. It would feed "
            "a governed IFRS 9 staging process rather than replace policy rules."
        ),
    }


def write_staging_outputs(
    staged: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
    summary_path: Path,
) -> None:
    """Persist staged rows and summary metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    staged.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Create the simplified IFRS 9 / SICR stage signal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path("data/model/scored_feature_matrix.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/staging/staged_scored_rows.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/staging_summary.json"),
    )
    parser.add_argument("--calibration-quantile", type=float, default=0.95)
    args = parser.parse_args()

    scored = load_scored_rows(args.scored)
    threshold = determine_sicr_threshold(scored, args.calibration_quantile)
    staged = assign_ifrs9_stage_signal(scored, threshold)
    summary = summarize_staging(staged, threshold, args.calibration_quantile)
    write_staging_outputs(staged, summary, args.output, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
