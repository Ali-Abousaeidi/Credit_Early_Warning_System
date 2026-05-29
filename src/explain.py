"""Officer-facing model explanation and reason-code helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.model import TARGET_COLUMN


def load_feature_columns(path: Path) -> list[str]:
    """Load the model feature-column order used during training."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_watchlist(path: Path) -> pd.DataFrame:
    """Load the committed watchlist rows with parsed dates."""
    return pd.read_csv(
        path,
        parse_dates=[
            "observation_month",
            "origination_month",
            "first_deterioration_month",
        ],
    )


def load_model(path: Path) -> XGBClassifier:
    """Load the trained XGBoost model artifact."""
    model = XGBClassifier()
    model.load_model(path)
    return model


def _coerce_model_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return numeric model inputs with booleans converted to integers."""
    matrix = frame[columns].copy()
    for column in matrix.columns:
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype(int)
    return matrix.astype(float)


def top_positive_contributors(
    shap_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return the largest positive SHAP contributors for one row."""
    contributors = [
        (feature, float(value))
        for feature, value in zip(feature_names, shap_row, strict=True)
        if value > 0
    ]
    contributors.sort(key=lambda item: item[1], reverse=True)
    return contributors[:top_n]


def format_reason(feature: str, value: float) -> str:
    """Convert a feature/value pair into a short officer-facing reason."""
    direction = "high" if value >= 0 else "low"
    if "utilization_slope" in feature:
        return f"Utilisation trend is rising ({feature}={value:.3f})"
    if feature == "utilization" or "utilization_mean" in feature:
        return f"Utilisation level is {direction} ({feature}={value:.3f})"
    if "payment_to_due_ratio_slope" in feature:
        return f"Payment-to-due trend is weakening ({feature}={value:.3f})"
    if feature == "payment_to_due_ratio" or "payment_to_due_ratio_mean" in feature:
        return f"Payment coverage is {direction} ({feature}={value:.3f})"
    if "payment_to_balance" in feature:
        return f"Payment relative to balance is weak ({feature}={value:.3f})"
    if "cash_buffer" in feature:
        return f"Cash buffer is thin or deteriorating ({feature}={value:.3f})"
    if "utilization_std" in feature:
        return f"Utilisation volatility is elevated ({feature}={value:.3f})"
    if "utilization_change" in feature:
        return f"Utilisation increased recently ({feature}={value:.3f})"
    if "missed_min_payment" in feature:
        return f"Recent minimum-payment stress ({feature}={value:.3f})"
    if "prior_30_plus" in feature or "months_since_last_30_plus" in feature:
        return f"Prior delinquency history contributes ({feature}={value:.3f})"
    if "balance_to_income" in feature:
        return f"Balance burden is {direction} ({feature}={value:.3f})"
    if "purchase_to_limit" in feature:
        return f"Recent usage activity is {direction} ({feature}={value:.3f})"
    return f"{feature} contributes to higher risk ({value:.3f})"


def build_reason_codes(
    watchlist: pd.DataFrame,
    model: XGBClassifier,
    feature_names: list[str],
    top_n: int = 3,
) -> pd.DataFrame:
    """Attach top SHAP reason codes to watchlist rows."""
    matrix = _coerce_model_matrix(watchlist, feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    rows: list[dict[str, object]] = []
    for row_idx, (_, row) in enumerate(watchlist.iterrows()):
        contributors = top_positive_contributors(shap_values[row_idx], feature_names, top_n)
        output: dict[str, object] = {
            "account_id": row["account_id"],
            "observation_month": row["observation_month"],
            "watchlist_rank": int(row["watchlist_rank"]),
            "predicted_probability": float(row["predicted_probability"]),
            TARGET_COLUMN: int(row[TARGET_COLUMN]),
            "months_to_deterioration": row["months_to_deterioration"],
        }
        for reason_idx in range(top_n):
            if reason_idx < len(contributors):
                feature, contribution = contributors[reason_idx]
                output[f"reason_feature_{reason_idx + 1}"] = feature
                output[f"reason_value_{reason_idx + 1}"] = float(row[feature])
                output[f"reason_shap_{reason_idx + 1}"] = contribution
                output[f"reason_{reason_idx + 1}"] = format_reason(feature, float(row[feature]))
            else:
                output[f"reason_feature_{reason_idx + 1}"] = ""
                output[f"reason_value_{reason_idx + 1}"] = np.nan
                output[f"reason_shap_{reason_idx + 1}"] = np.nan
                output[f"reason_{reason_idx + 1}"] = ""
        rows.append(output)
    return pd.DataFrame(rows)


def summarize_reason_codes(reason_codes: pd.DataFrame) -> dict[str, object]:
    """Summarize the most common primary reason features."""
    top_features = (
        reason_codes["reason_feature_1"].value_counts().head(10).to_dict()
        if not reason_codes.empty
        else {}
    )
    return {
        "n_watchlist_rows_explained": int(len(reason_codes)),
        "average_predicted_probability": round(
            float(reason_codes["predicted_probability"].mean()), 4
        )
        if not reason_codes.empty
        else 0.0,
        "top_primary_reason_features": {
            key: int(value) for key, value in top_features.items()
        },
    }


def save_reason_frequency_plot(reason_codes: pd.DataFrame, figures_dir: Path) -> None:
    """Save a bar chart of the most common primary reason features."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = reason_codes["reason_feature_1"].value_counts().head(10).sort_values()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="barh", color="#176D8C")
    plt.xlabel("Watchlist rows where feature is primary reason")
    plt.ylabel("Primary reason feature")
    plt.title("Most Common Primary SHAP Reason Features")
    plt.tight_layout()
    plt.savefig(figures_dir / "reason_feature_frequency.png", dpi=160)
    plt.close()


def write_reason_outputs(
    reason_codes: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
    summary_path: Path,
    figures_dir: Path,
) -> None:
    """Persist reason-code table, summary, and frequency plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    reason_codes.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_reason_frequency_plot(reason_codes, figures_dir)


def main() -> None:
    """Generate SHAP reason codes for watchlist rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/ews_xgb_model.json"))
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=Path("reports/tables/watchlist_top100_test.csv"),
    )
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=Path("reports/tables/feature_columns.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/watchlist_reason_codes.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/reason_code_summary.json"),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    model = load_model(args.model)
    watchlist = load_watchlist(args.watchlist)
    columns = load_feature_columns(args.feature_columns)
    reason_codes = build_reason_codes(watchlist, model, columns)
    summary = summarize_reason_codes(reason_codes)
    write_reason_outputs(reason_codes, summary, args.output, args.summary, args.figures_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
