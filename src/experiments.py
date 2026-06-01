"""Benchmark models, feature ablations, and review-capacity sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import RANDOM_SEED
from src.features import feature_columns
from src.model import ModelConfig, TARGET_COLUMN, add_time_split, _coerce_model_matrix


def is_trend_feature(column: str) -> bool:
    """Return whether a model feature is an engineered behavioural trend feature."""
    return (
        column.endswith(("_slope_3m", "_slope_6m", "_change_1m", "_std_3m", "_std_6m"))
        or column.startswith(("missed_min_payment_count_", "missed_min_payment_rate_"))
        or column.startswith("first_missed_min_payment_")
        or column in {"prior_30_plus_count_12m", "months_since_last_30_plus"}
    )


def level_feature_columns(columns: list[str]) -> list[str]:
    """Return static/current-level features, excluding engineered trend columns."""
    return [column for column in columns if not is_trend_feature(column)]


def trend_feature_columns(columns: list[str]) -> list[str]:
    """Return engineered trend/window features."""
    return [column for column in columns if is_trend_feature(column)]


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load feature matrix with dates parsed."""
    return pd.read_csv(
        path,
        parse_dates=["observation_month", "origination_month", "first_deterioration_month"],
    )


def evaluate_scores(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    """Evaluate probability-like risk scores."""
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, clipped)), 4),
        "pr_auc": round(float(average_precision_score(y_true, clipped)), 4),
        "brier_score": round(float(brier_score_loss(y_true, clipped)), 4),
    }


def xgb_model(scale_pos_weight: float, seed: int = RANDOM_SEED) -> XGBClassifier:
    """Create a compact XGBoost model for benchmarks and ablations."""
    return XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )


def train_predict_xgb(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int
) -> np.ndarray:
    """Train XGBoost and return test probabilities."""
    y_train = train[TARGET_COLUMN].astype(int)
    scale_pos_weight = (len(y_train) - int(y_train.sum())) / int(y_train.sum())
    model = xgb_model(scale_pos_weight=scale_pos_weight, seed=seed)
    model.fit(_coerce_model_matrix(train, columns), y_train)
    return model.predict_proba(_coerce_model_matrix(test, columns))[:, 1]


def benchmark_models(split_frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compare production-style model against simple, defensible baselines."""
    train = split_frame.loc[split_frame["split"].eq("train")]
    test = split_frame.loc[split_frame["split"].eq("test")]
    y_train = train[TARGET_COLUMN].astype(int)
    y_test = test[TARGET_COLUMN].astype(int)
    X_train = _coerce_model_matrix(train, columns)
    X_test = _coerce_model_matrix(test, columns)

    results: list[dict[str, object]] = []

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=RANDOM_SEED,
        ),
    )
    logistic.fit(X_train, y_train)
    results.append(
        {
            "experiment": "benchmark",
            "model": "logistic_regression_balanced",
            "feature_set": "full",
            **evaluate_scores(y_test, logistic.predict_proba(X_test)[:, 1]),
        }
    )

    forest = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=40,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    forest.fit(X_train, y_train)
    results.append(
        {
            "experiment": "benchmark",
            "model": "random_forest_balanced",
            "feature_set": "full",
            **evaluate_scores(y_test, forest.predict_proba(X_test)[:, 1]),
        }
    )

    xgb_scores = train_predict_xgb(train, test, columns, RANDOM_SEED)
    results.append(
        {
            "experiment": "benchmark",
            "model": "xgboost",
            "feature_set": "full",
            **evaluate_scores(y_test, xgb_scores),
        }
    )

    rule_score = (
        test["utilization_change_1m"].rank(pct=True)
        + test["utilization_std_3m"].rank(pct=True)
        + (-test["payment_to_due_ratio_slope_3m"]).rank(pct=True)
        + (-test["cash_buffer_months_change_1m"]).rank(pct=True)
    ) / 4
    results.append(
        {
            "experiment": "benchmark",
            "model": "transparent_rule_score",
            "feature_set": "four_behavioural_rules",
            **evaluate_scores(y_test, rule_score.to_numpy()),
        }
    )
    return pd.DataFrame(results)


def ablation_study(split_frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compare full, level-only, and trend-only XGBoost models."""
    train = split_frame.loc[split_frame["split"].eq("train")]
    test = split_frame.loc[split_frame["split"].eq("test")]
    y_test = test[TARGET_COLUMN].astype(int)
    feature_sets = {
        "full": columns,
        "level_only": level_feature_columns(columns),
        "trend_only": trend_feature_columns(columns),
    }
    rows: list[dict[str, object]] = []
    for name, subset in feature_sets.items():
        scores = train_predict_xgb(train, test, subset, RANDOM_SEED)
        rows.append(
            {
                "experiment": "ablation",
                "model": "xgboost",
                "feature_set": name,
                "n_features": len(subset),
                **evaluate_scores(y_test, scores),
            }
        )
    return pd.DataFrame(rows)


def capacity_sensitivity(scored: pd.DataFrame, capacities: list[int]) -> pd.DataFrame:
    """Evaluate precision, account capture, and lead time by review capacity."""
    from src.watchlist import account_level_lead_times, build_monthly_watchlist

    test = scored.loc[scored["split"].eq("test")].copy()
    positive_accounts = set(test.loc[test[TARGET_COLUMN].eq(1), "account_id"])
    rows: list[dict[str, object]] = []
    for capacity in capacities:
        watchlist = build_monthly_watchlist(test, split_name="test", top_k_per_month=capacity)
        lead_times = account_level_lead_times(watchlist)
        rows.append(
            {
                "top_k_per_month": capacity,
                "watchlist_rows": int(len(watchlist)),
                "precision_at_k": round(float(watchlist[TARGET_COLUMN].mean()), 4),
                "account_capture_rate": round(
                    float(lead_times["account_id"].nunique() / len(positive_accounts)), 4
                ),
                "median_lead_time_months": round(
                    float(lead_times["lead_time_months"].median()), 2
                )
                if not lead_times.empty
                else 0.0,
                "mean_lead_time_months": round(float(lead_times["lead_time_months"].mean()), 2)
                if not lead_times.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_experiment_plots(
    experiments: pd.DataFrame, capacity: pd.DataFrame, figures_dir: Path
) -> None:
    """Save benchmark/ablation and capacity-sensitivity plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plot_data = experiments.sort_values("pr_auc")
    labels = plot_data["model"] + " / " + plot_data["feature_set"]
    plt.barh(labels, plot_data["pr_auc"], color="#176D8C")
    plt.xlabel("Out-of-time PR-AUC")
    plt.title("Benchmark and Ablation Comparison")
    plt.tight_layout()
    plt.savefig(figures_dir / "benchmark_ablation_pr_auc.png", dpi=160)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(
        capacity["top_k_per_month"],
        capacity["precision_at_k"],
        marker="o",
        color="#176D8C",
        label="Precision@k",
    )
    ax1.set_xlabel("Monthly review capacity")
    ax1.set_ylabel("Precision@k")
    ax2 = ax1.twinx()
    ax2.plot(
        capacity["top_k_per_month"],
        capacity["account_capture_rate"],
        marker="s",
        color="#8A4F7D",
        label="Account capture",
    )
    ax2.set_ylabel("Account capture rate")
    ax1.set_title("Watchlist Capacity Sensitivity")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    plt.tight_layout()
    plt.savefig(figures_dir / "capacity_sensitivity.png", dpi=160)
    plt.close()


def summarize_experiments(
    experiments: pd.DataFrame, capacity: pd.DataFrame
) -> dict[str, object]:
    """Summarize benchmark, ablation, and capacity sensitivity results."""
    best = experiments.sort_values("pr_auc", ascending=False).iloc[0]
    full = experiments.loc[
        experiments["experiment"].eq("ablation") & experiments["feature_set"].eq("full")
    ].iloc[0]
    trend = experiments.loc[
        experiments["experiment"].eq("ablation") & experiments["feature_set"].eq("trend_only")
    ].iloc[0]
    level = experiments.loc[
        experiments["experiment"].eq("ablation") & experiments["feature_set"].eq("level_only")
    ].iloc[0]
    return {
        "best_pr_auc_model": {
            "model": best["model"],
            "feature_set": best["feature_set"],
            "pr_auc": round(float(best["pr_auc"]), 4),
        },
        "ablation_pr_auc": {
            "full": round(float(full["pr_auc"]), 4),
            "level_only": round(float(level["pr_auc"]), 4),
            "trend_only": round(float(trend["pr_auc"]), 4),
        },
        "capacity_rows": capacity.to_dict(orient="records"),
    }


def main() -> None:
    """Run benchmark, ablation, and capacity-sensitivity experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/feature_matrix.csv"),
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path("data/staging/staged_scored_rows.csv"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    feature_matrix = load_feature_matrix(args.features)
    split_frame = add_time_split(feature_matrix, ModelConfig())
    columns = feature_columns(split_frame)
    experiments = pd.concat(
        [benchmark_models(split_frame, columns), ablation_study(split_frame, columns)],
        ignore_index=True,
    )
    scored = pd.read_csv(
        args.scored,
        parse_dates=["observation_month", "origination_month", "first_deterioration_month"],
    )
    capacity = capacity_sensitivity(scored, capacities=[25, 50, 100, 150, 200])
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    experiments.to_csv(args.reports_dir / "benchmark_ablation_results.csv", index=False)
    capacity.to_csv(args.reports_dir / "capacity_sensitivity.csv", index=False)
    summary = summarize_experiments(experiments, capacity)
    (args.reports_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_experiment_plots(experiments, capacity, args.figures_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
