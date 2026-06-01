"""Auxiliary time-to-deterioration model for positive early-warning cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error

from src import RANDOM_SEED
from src.features import feature_columns
from src.model import TARGET_COLUMN, ModelConfig, add_time_split, _coerce_model_matrix


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load feature matrix with dates parsed."""
    return pd.read_csv(
        path,
        parse_dates=["observation_month", "origination_month", "first_deterioration_month"],
    )


def fit_time_to_event_model(train_positive: pd.DataFrame, columns: list[str]) -> RandomForestRegressor:
    """Fit a simple positive-case model for months until deterioration."""
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(
        _coerce_model_matrix(train_positive, columns),
        train_positive["months_to_deterioration"].astype(float),
    )
    return model


def run_time_to_event(feature_matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train and evaluate the auxiliary time-to-event model."""
    split_frame = add_time_split(feature_matrix, ModelConfig())
    columns = feature_columns(split_frame)
    train_positive = split_frame.loc[
        split_frame["split"].isin(["train", "calibration"])
        & split_frame[TARGET_COLUMN].eq(1)
    ].copy()
    test_positive = split_frame.loc[
        split_frame["split"].eq("test") & split_frame[TARGET_COLUMN].eq(1)
    ].copy()
    model = fit_time_to_event_model(train_positive, columns)
    predictions = model.predict(_coerce_model_matrix(test_positive, columns))
    predictions = np.clip(predictions, 1.0, 6.0)
    result = test_positive[
        ["account_id", "observation_month", "months_to_deterioration"]
    ].copy()
    result["predicted_months_to_deterioration"] = predictions
    result["absolute_error_months"] = (
        result["predicted_months_to_deterioration"]
        - result["months_to_deterioration"].astype(float)
    ).abs()
    summary = {
        "train_positive_rows": int(len(train_positive)),
        "test_positive_rows": int(len(test_positive)),
        "mean_absolute_error_months": round(
            float(mean_absolute_error(result["months_to_deterioration"], predictions)), 3
        ),
        "median_absolute_error_months": round(
            float(median_absolute_error(result["months_to_deterioration"], predictions)), 3
        ),
        "prediction_mean_months": round(float(result["predicted_months_to_deterioration"].mean()), 3),
        "actual_mean_months": round(float(result["months_to_deterioration"].mean()), 3),
        "note": (
            "This is an auxiliary positive-case time-to-event model, not a full "
            "censored survival model."
        ),
    }
    return result, summary


def save_time_to_event_plot(results: pd.DataFrame, figures_dir: Path) -> None:
    """Save actual-vs-predicted months-to-deterioration plot."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(
        results["months_to_deterioration"],
        results["predicted_months_to_deterioration"],
        alpha=0.35,
        color="#176D8C",
        edgecolors="none",
    )
    plt.plot([1, 6], [1, 6], color="gray", linestyle=":")
    plt.xlabel("Actual months to deterioration")
    plt.ylabel("Predicted months to deterioration")
    plt.title("Auxiliary Time-to-Deterioration Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "time_to_event_actual_vs_predicted.png", dpi=160)
    plt.close()


def main() -> None:
    """Run auxiliary time-to-event modelling for positive cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/feature_matrix.csv"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    feature_matrix = load_feature_matrix(args.features)
    results, summary = run_time_to_event(feature_matrix)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.reports_dir / "time_to_event_predictions.csv", index=False)
    (args.reports_dir / "time_to_event_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_time_to_event_plot(results, args.figures_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
