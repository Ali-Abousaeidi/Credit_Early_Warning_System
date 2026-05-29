"""Early-warning classifier and calibration workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src import RANDOM_SEED
from src.features import feature_columns
from src.target import WARNING_HORIZON_MONTHS

TARGET_COLUMN = f"target_deterioration_{WARNING_HORIZON_MONTHS}m"


@dataclass(frozen=True)
class ModelConfig:
    """Time split and model hyperparameters for Phase 5."""

    train_end: str = "2021-06-30"
    calibration_end: str = "2021-12-31"
    seed: int = RANDOM_SEED
    n_estimators: int = 250
    max_depth: int = 3
    learning_rate: float = 0.04
    subsample: float = 0.85
    colsample_bytree: float = 0.85


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load the generated feature matrix with parsed dates."""
    return pd.read_csv(
        path,
        parse_dates=[
            "observation_month",
            "origination_month",
            "first_deterioration_month",
        ],
    )


def add_time_split(feature_matrix: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Assign train/calibration/test splits by observation month."""
    split_frame = feature_matrix.copy()
    train_end = pd.Timestamp(config.train_end)
    calibration_end = pd.Timestamp(config.calibration_end)
    if train_end >= calibration_end:
        raise ValueError("train_end must be before calibration_end.")

    split_frame["split"] = np.select(
        [
            split_frame["observation_month"].le(train_end),
            split_frame["observation_month"].gt(train_end)
            & split_frame["observation_month"].le(calibration_end),
            split_frame["observation_month"].gt(calibration_end),
        ],
        ["train", "calibration", "test"],
        default="unassigned",
    )
    if split_frame["split"].eq("unassigned").any():
        raise ValueError("Some rows were not assigned to a time split.")
    return split_frame


def _coerce_model_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return numeric model inputs with booleans converted to integers."""
    matrix = frame[columns].copy()
    for column in matrix.columns:
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype(int)
    return matrix.astype(float)


def train_xgboost_classifier(
    train_frame: pd.DataFrame, columns: list[str], config: ModelConfig
) -> XGBClassifier:
    """Fit an imbalanced XGBoost classifier on the training window."""
    y_train = train_frame[TARGET_COLUMN].astype(int)
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    if positive == 0:
        raise ValueError("Training window has no positive target rows.")
    scale_pos_weight = negative / positive

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.seed,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(_coerce_model_matrix(train_frame, columns), y_train)
    return model


def fit_isotonic_calibrator(
    model: XGBClassifier, calibration_frame: pd.DataFrame, columns: list[str]
) -> IsotonicRegression:
    """Fit probability calibration on the calibration window only."""
    raw_probabilities = model.predict_proba(
        _coerce_model_matrix(calibration_frame, columns)
    )[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_probabilities, calibration_frame[TARGET_COLUMN].astype(int))
    return calibrator


def score_frame(
    model: XGBClassifier,
    calibrator: IsotonicRegression,
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add raw and calibrated deterioration probabilities."""
    scored = frame.copy()
    raw_probability = model.predict_proba(_coerce_model_matrix(scored, columns))[:, 1]
    scored["raw_probability"] = raw_probability
    scored["predicted_probability"] = calibrator.transform(raw_probability)
    return scored


def evaluate_scored_frame(scored_frame: pd.DataFrame) -> dict[str, float]:
    """Calculate discrimination and calibration metrics."""
    y_true = scored_frame[TARGET_COLUMN].astype(int)
    y_score = scored_frame["predicted_probability"]
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_score)), 4),
        "log_loss": round(float(log_loss(y_true, y_score, labels=[0, 1])), 4),
        "base_rate": round(float(y_true.mean()), 4),
    }


def summarize_model_run(
    scored: pd.DataFrame, columns: list[str], config: ModelConfig
) -> dict[str, object]:
    """Summarize split sizes, target rates, and out-of-time metrics."""
    split_summary: dict[str, object] = {}
    for split_name, split_frame in scored.groupby("split", sort=False):
        split_summary[split_name] = {
            "rows": int(len(split_frame)),
            "positive_rows": int(split_frame[TARGET_COLUMN].sum()),
            "positive_rate": round(float(split_frame[TARGET_COLUMN].mean()), 4),
            "month_min": str(split_frame["observation_month"].min().date()),
            "month_max": str(split_frame["observation_month"].max().date()),
        }
    test_metrics = evaluate_scored_frame(scored.loc[scored["split"].eq("test")])
    return {
        "config": asdict(config),
        "target_column": TARGET_COLUMN,
        "n_model_features": len(columns),
        "split_summary": split_summary,
        "test_metrics": test_metrics,
    }


def save_precision_recall_plot(scored_test: pd.DataFrame, figures_dir: Path) -> None:
    """Save the out-of-time precision-recall curve."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(
        scored_test[TARGET_COLUMN].astype(int), scored_test["predicted_probability"]
    )
    pr_auc = average_precision_score(
        scored_test[TARGET_COLUMN].astype(int), scored_test["predicted_probability"]
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}", color="#176D8C")
    plt.axhline(
        scored_test[TARGET_COLUMN].mean(),
        color="gray",
        linestyle=":",
        label="Base rate",
    )
    plt.xlabel("Recall / capture rate")
    plt.ylabel("Precision")
    plt.title("Out-of-Time Early-Warning Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_precision_recall.png", dpi=160)
    plt.close()


def save_calibration_plot(scored_test: pd.DataFrame, figures_dir: Path) -> None:
    """Save a decile calibration plot for the out-of-time test window."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    calibration = scored_test.copy()
    calibration["score_decile"] = pd.qcut(
        calibration["predicted_probability"].rank(method="first"),
        q=10,
        labels=False,
    )
    deciles = (
        calibration.groupby("score_decile")
        .agg(
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_deterioration_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index(drop=True)
    )

    plt.figure(figsize=(6, 5))
    plt.plot(
        deciles["mean_predicted_probability"],
        deciles["observed_deterioration_rate"],
        marker="o",
        color="#8A4F7D",
    )
    max_axis = float(
        max(
            deciles["mean_predicted_probability"].max(),
            deciles["observed_deterioration_rate"].max(),
        )
    )
    plt.plot([0, max_axis], [0, max_axis], color="gray", linestyle=":")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed deterioration rate")
    plt.title("Out-of-Time Calibration by Score Decile")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_calibration.png", dpi=160)
    plt.close()


def write_model_outputs(
    model: XGBClassifier,
    calibrator: IsotonicRegression,
    scored: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
    scores_path: Path,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Persist model artifacts, scored rows, metrics, and figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "ews_xgb_model.json")
    pd.DataFrame(
        {
            "x_thresholds": calibrator.X_thresholds_,
            "y_thresholds": calibrator.y_thresholds_,
        }
    ).to_csv(output_dir / "isotonic_calibration.csv", index=False)
    scored.to_csv(scores_path, index=False)
    (reports_dir / "model_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    scored.loc[scored["split"].eq("test"), [
        "account_id",
        "observation_month",
        TARGET_COLUMN,
        "months_to_deterioration",
        "first_deterioration_month",
        "predicted_probability",
        "raw_probability",
    ]].to_csv(reports_dir / "test_predictions.csv", index=False)
    scored_test = scored.loc[scored["split"].eq("test")]
    save_precision_recall_plot(scored_test, figures_dir)
    save_calibration_plot(scored_test, figures_dir)


def run_training(feature_matrix: pd.DataFrame, config: ModelConfig) -> tuple[pd.DataFrame, dict[str, object], XGBClassifier, IsotonicRegression]:
    """Train, calibrate, score, and summarize the early-warning model."""
    split_frame = add_time_split(feature_matrix, config)
    columns = feature_columns(split_frame)
    train_frame = split_frame.loc[split_frame["split"].eq("train")]
    calibration_frame = split_frame.loc[split_frame["split"].eq("calibration")]
    model = train_xgboost_classifier(train_frame, columns, config)
    calibrator = fit_isotonic_calibrator(model, calibration_frame, columns)
    scored = score_frame(model, calibrator, split_frame, columns)
    summary = summarize_model_run(scored, columns, config)
    return scored, summary, model, calibrator


def main() -> None:
    """Train and evaluate the Phase 5 early-warning model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/feature_matrix.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("data/model/scored_feature_matrix.csv"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    feature_matrix = load_feature_matrix(args.features)
    config = ModelConfig()
    scored, summary, model, calibrator = run_training(feature_matrix, config)
    write_model_outputs(
        model=model,
        calibrator=calibrator,
        scored=scored,
        summary=summary,
        output_dir=args.output_dir,
        scores_path=args.scores_output,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
