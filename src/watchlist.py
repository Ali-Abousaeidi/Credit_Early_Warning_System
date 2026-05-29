"""Ranked watchlist and lead-time analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.model import TARGET_COLUMN


def load_staged_scores(path: Path) -> pd.DataFrame:
    """Load staged and scored rows with parsed dates."""
    return pd.read_csv(
        path,
        parse_dates=[
            "observation_month",
            "origination_month",
            "first_deterioration_month",
        ],
    )


def build_monthly_watchlist(
    scored: pd.DataFrame,
    split_name: str = "test",
    top_k_per_month: int = 100,
) -> pd.DataFrame:
    """Return the top-k highest-risk rows per observation month."""
    if top_k_per_month < 1:
        raise ValueError("top_k_per_month must be positive.")
    subset = scored.loc[scored["split"].eq(split_name)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split: {split_name}")
    watchlist = (
        subset.sort_values(
            ["observation_month", "predicted_probability"],
            ascending=[True, False],
        )
        .groupby("observation_month", group_keys=False)
        .head(top_k_per_month)
        .copy()
    )
    watchlist["watchlist_rank"] = (
        watchlist.groupby("observation_month")["predicted_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return watchlist.sort_values(["observation_month", "watchlist_rank"]).reset_index(
        drop=True
    )


def account_level_lead_times(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Calculate earliest warning lead time for flagged deteriorating accounts."""
    hits = watchlist.loc[watchlist[TARGET_COLUMN].eq(1)].copy()
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "account_id",
                "first_deterioration_month",
                "first_flag_month",
                "lead_time_months",
                "max_predicted_probability",
            ]
        )
    account_hits = (
        hits.sort_values(["account_id", "months_to_deterioration"], ascending=[True, False])
        .groupby("account_id", as_index=False)
        .first()
    )
    account_hits = account_hits.rename(
        columns={
            "observation_month": "first_flag_month",
            "months_to_deterioration": "lead_time_months",
            "predicted_probability": "max_predicted_probability",
        }
    )
    return account_hits[
        [
            "account_id",
            "first_deterioration_month",
            "first_flag_month",
            "lead_time_months",
            "max_predicted_probability",
        ]
    ]


def summarize_watchlist(
    scored: pd.DataFrame,
    watchlist: pd.DataFrame,
    lead_times: pd.DataFrame,
    split_name: str,
    top_k_per_month: int,
) -> dict[str, object]:
    """Summarize precision, capture, and lead-time value."""
    test = scored.loc[scored["split"].eq(split_name)].copy()
    positive_accounts = set(test.loc[test[TARGET_COLUMN].eq(1), "account_id"])
    flagged_positive_accounts = set(lead_times["account_id"])
    monthly_precision = (
        watchlist.groupby("observation_month")[TARGET_COLUMN]
        .mean()
        .rename("precision_at_k")
    )
    monthly_hits = watchlist.groupby("observation_month")[TARGET_COLUMN].sum()
    return {
        "split": split_name,
        "top_k_per_month": top_k_per_month,
        "n_watchlist_rows": int(len(watchlist)),
        "n_watchlist_months": int(watchlist["observation_month"].nunique()),
        "watchlist_positive_rows": int(watchlist[TARGET_COLUMN].sum()),
        "precision_at_k_overall": round(float(watchlist[TARGET_COLUMN].mean()), 4),
        "precision_at_k_monthly_mean": round(float(monthly_precision.mean()), 4),
        "monthly_positive_hits": {
            str(month.date()): int(value) for month, value in monthly_hits.items()
        },
        "positive_accounts_in_split": int(len(positive_accounts)),
        "positive_accounts_flagged": int(len(flagged_positive_accounts)),
        "account_capture_rate": round(
            len(flagged_positive_accounts) / len(positive_accounts), 4
        )
        if positive_accounts
        else 0.0,
        "median_lead_time_months": round(float(lead_times["lead_time_months"].median()), 2)
        if not lead_times.empty
        else 0.0,
        "mean_lead_time_months": round(float(lead_times["lead_time_months"].mean()), 2)
        if not lead_times.empty
        else 0.0,
        "lead_time_distribution": {
            str(int(months)): int(count)
            for months, count in lead_times["lead_time_months"].value_counts().sort_index().items()
        },
    }


def save_lead_time_plot(lead_times: pd.DataFrame, figures_dir: Path) -> None:
    """Save the lead-time distribution plot."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.8))
    if lead_times.empty:
        plt.text(0.5, 0.5, "No flagged deteriorating accounts", ha="center", va="center")
        plt.axis("off")
    else:
        counts = lead_times["lead_time_months"].value_counts().sort_index()
        plt.bar(counts.index.astype(int), counts.values, color="#176D8C")
        plt.xlabel("Months between first flag and first 30+ DPD")
        plt.ylabel("Flagged deteriorating accounts")
        plt.title("Lead Time Distribution for Watchlist Hits")
        plt.xticks(counts.index.astype(int))
    plt.tight_layout()
    plt.savefig(figures_dir / "lead_time.png", dpi=160)
    plt.close()


def write_watchlist_outputs(
    watchlist: pd.DataFrame,
    lead_times: pd.DataFrame,
    summary: dict[str, object],
    watchlist_path: Path,
    lead_time_path: Path,
    summary_path: Path,
    figures_dir: Path,
) -> None:
    """Persist watchlist, account lead times, summary, and plot."""
    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    lead_time_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(watchlist_path, index=False)
    lead_times.to_csv(lead_time_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_lead_time_plot(lead_times, figures_dir)


def run_watchlist(
    scored: pd.DataFrame,
    split_name: str = "test",
    top_k_per_month: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build watchlist and calculate lead-time metrics."""
    watchlist = build_monthly_watchlist(scored, split_name, top_k_per_month)
    lead_times = account_level_lead_times(watchlist)
    summary = summarize_watchlist(scored, watchlist, lead_times, split_name, top_k_per_month)
    return watchlist, lead_times, summary


def main() -> None:
    """Build the ranked watchlist and lead-time analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path("data/staging/staged_scored_rows.csv"),
    )
    parser.add_argument("--top-k-per-month", type=int, default=100)
    parser.add_argument(
        "--watchlist-output",
        type=Path,
        default=Path("reports/tables/watchlist_top100_test.csv"),
    )
    parser.add_argument(
        "--lead-times-output",
        type=Path,
        default=Path("reports/tables/watchlist_lead_times.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/tables/watchlist_summary.json"),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    scored = load_staged_scores(args.scored)
    watchlist, lead_times, summary = run_watchlist(
        scored,
        split_name="test",
        top_k_per_month=args.top_k_per_month,
    )
    write_watchlist_outputs(
        watchlist=watchlist,
        lead_times=lead_times,
        summary=summary,
        watchlist_path=args.watchlist_output,
        lead_time_path=args.lead_times_output,
        summary_path=args.summary,
        figures_dir=args.figures_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
