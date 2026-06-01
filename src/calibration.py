"""Calibration comparison for early-warning probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.model import TARGET_COLUMN


def expected_calibration_error(
    y_true: pd.Series, y_score: pd.Series, n_bins: int = 10
) -> float:
    """Calculate equal-width expected calibration error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1], right=True)
    total = len(y_true)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        observed = float(y_true[mask].mean())
        predicted = float(y_score[mask].mean())
        ece += (mask.sum() / total) * abs(observed - predicted)
    return float(ece)


def fit_platt_scores(scored: pd.DataFrame) -> pd.Series:
    """Fit Platt scaling on calibration rows and score every row."""
    calibration = scored.loc[scored["split"].eq("calibration")]
    model = LogisticRegression(random_state=42)
    model.fit(
        calibration[["raw_probability"]],
        calibration[TARGET_COLUMN].astype(int),
    )
    return pd.Series(
        model.predict_proba(scored[["raw_probability"]])[:, 1],
        index=scored.index,
        name="platt_probability",
    )


def compare_calibration(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw, isotonic, and Platt-calibrated probabilities."""
    compared = scored.copy()
    compared["platt_probability"] = fit_platt_scores(compared)
    test = compared.loc[compared["split"].eq("test")].copy()
    y_true = test[TARGET_COLUMN].astype(int)
    probability_columns = {
        "raw_xgboost": "raw_probability",
        "isotonic": "predicted_probability",
        "platt": "platt_probability",
    }
    rows: list[dict[str, float | str]] = []
    for name, column in probability_columns.items():
        probabilities = test[column].clip(1e-6, 1 - 1e-6)
        rows.append(
            {
                "calibration_method": name,
                "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
                "log_loss": round(float(log_loss(y_true, probabilities)), 4),
                "expected_calibration_error": round(
                    expected_calibration_error(y_true, probabilities),
                    4,
                ),
                "mean_predicted_probability": round(float(probabilities.mean()), 4),
                "observed_rate": round(float(y_true.mean()), 4),
            }
        )
    return compared, pd.DataFrame(rows)


def save_calibration_comparison_plot(results: pd.DataFrame, figures_dir: Path) -> None:
    """Save comparison of Brier score and ECE by calibration method."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_data = results.set_index("calibration_method")[
        ["brier_score", "expected_calibration_error"]
    ]
    ax = plot_data.plot(kind="bar", figsize=(7, 4.8), color=["#176D8C", "#8A4F7D"])
    ax.set_xlabel("Calibration method")
    ax.set_ylabel("Lower is better")
    ax.set_title("Out-of-Time Calibration Comparison")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_comparison.png", dpi=160)
    plt.close()


def main() -> None:
    """Run calibration comparison from scored model rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path("data/model/scored_feature_matrix.csv"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    scored = pd.read_csv(
        args.scored,
        parse_dates=["observation_month", "origination_month", "first_deterioration_month"],
    )
    compared, results = compare_calibration(scored)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.reports_dir / "calibration_comparison.csv", index=False)
    (args.reports_dir / "calibration_comparison_summary.json").write_text(
        json.dumps(results.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    save_calibration_comparison_plot(results, args.figures_dir)
    print(json.dumps(results.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
