"""Exploratory transition behaviour for the monitoring panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

STATUS_ORDER: Final[list[str]] = ["current", "dpd_30", "dpd_60", "dpd_90", "default"]


def load_panel(path: Path) -> pd.DataFrame:
    """Load the generated account-month panel with parsed dates."""
    return pd.read_csv(
        path,
        parse_dates=["observation_month", "origination_month"],
    )


def calculate_transition_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Count one-month performance-status transitions by account."""
    ordered = panel.sort_values(["account_id", "observation_month"]).copy()
    ordered["next_status"] = ordered.groupby("account_id")["performance_status"].shift(-1)
    transitions = ordered.dropna(subset=["next_status"])
    counts = pd.crosstab(
        transitions["performance_status"],
        transitions["next_status"],
    )
    return counts.reindex(index=STATUS_ORDER, columns=STATUS_ORDER, fill_value=0)


def calculate_transition_probabilities(counts: pd.DataFrame) -> pd.DataFrame:
    """Convert transition counts to row-normalized probabilities."""
    row_totals = counts.sum(axis=1).replace(0, pd.NA)
    return counts.div(row_totals, axis=0).fillna(0.0)


def first_deterioration_months(panel: pd.DataFrame) -> pd.Series:
    """Return first observed 30+ DPD month for each account that deteriorates."""
    deteriorated = panel.loc[panel["days_past_due"] >= 30]
    return deteriorated.groupby("account_id")["observation_month"].min()


def save_transition_outputs(
    counts: pd.DataFrame,
    probabilities: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    """Persist transition tables and a probability heatmap."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    counts.to_csv(tables_dir / "transition_counts.csv")
    probabilities.to_csv(tables_dir / "transition_probabilities.csv")

    plt.figure(figsize=(8, 5.5))
    sns.heatmap(
        probabilities,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "One-month transition probability"},
    )
    plt.title("One-Month Performance Status Transition Matrix")
    plt.xlabel("Next month status")
    plt.ylabel("Current month status")
    plt.tight_layout()
    plt.savefig(figures_dir / "transition_matrix.png", dpi=160)
    plt.close()


def save_example_trajectory_plot(panel: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Plot example deterioration trajectories around first 30+ DPD event."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    first_events = first_deterioration_months(panel)
    if first_events.empty:
        raise ValueError("No deteriorating accounts found for trajectory plot.")

    chosen_accounts = first_events.sort_values().head(6).index.tolist()
    examples = panel.loc[panel["account_id"].isin(chosen_accounts)].copy()
    event_map = first_events.loc[chosen_accounts].to_dict()
    examples["event_month"] = examples["account_id"].map(event_map)
    examples["months_from_first_30_dpd"] = (
        (examples["observation_month"].dt.year - examples["event_month"].dt.year) * 12
        + examples["observation_month"].dt.month
        - examples["event_month"].dt.month
    )
    examples = examples.loc[examples["months_from_first_30_dpd"].between(-8, 4)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    sns.lineplot(
        data=examples,
        x="months_from_first_30_dpd",
        y="utilization",
        hue="account_id",
        marker="o",
        ax=axes[0],
        legend=False,
    )
    axes[0].axvline(0, color="firebrick", linestyle="--", linewidth=1)
    axes[0].set_title("Example Deterioration Trajectories")
    axes[0].set_ylabel("Utilization")

    sns.lineplot(
        data=examples,
        x="months_from_first_30_dpd",
        y="payment_to_due_ratio",
        hue="account_id",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1)
    axes[1].axvline(0, color="firebrick", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Months from first 30+ DPD observation")
    axes[1].set_ylabel("Payment / scheduled due")

    plt.tight_layout()
    plt.savefig(figures_dir / "example_deterioration_trajectories.png", dpi=160)
    plt.close()
    return chosen_accounts


def summarize_transition_analysis(
    panel: pd.DataFrame,
    counts: pd.DataFrame,
    probabilities: pd.DataFrame,
    chosen_accounts: list[str],
) -> dict[str, object]:
    """Summarize Phase 2 EDA outputs for README/reporting."""
    first_events = first_deterioration_months(panel)
    performing_rows = counts.loc["current"].sum()
    current_to_30_plus = counts.loc["current", ["dpd_30", "dpd_60", "dpd_90", "default"]].sum()
    status_mix = panel["performance_status"].value_counts(normalize=True).sort_index()
    return {
        "n_one_month_transitions": int(counts.to_numpy().sum()),
        "performing_to_30_plus_count": int(current_to_30_plus),
        "performing_to_30_plus_rate": round(float(current_to_30_plus / performing_rows), 4),
        "current_roll_rate": round(float(probabilities.loc["current", "current"]), 4),
        "dpd_30_cure_rate": round(float(probabilities.loc["dpd_30", "current"]), 4),
        "dpd_30_to_worse_rate": round(
            float(probabilities.loc["dpd_30", ["dpd_60", "dpd_90", "default"]].sum()),
            4,
        ),
        "n_accounts_with_first_30_plus_dpd": int(first_events.shape[0]),
        "status_mix": {key: round(float(value), 4) for key, value in status_mix.items()},
        "example_accounts": chosen_accounts,
    }


def main() -> None:
    """Run Phase 2 transition EDA and write tables/figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/panel/account_month_panel.csv"),
        help="Path to the account-month panel generated by src.data_panel.",
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    panel = load_panel(args.panel)
    counts = calculate_transition_counts(panel)
    probabilities = calculate_transition_probabilities(counts)
    save_transition_outputs(counts, probabilities, args.tables_dir, args.figures_dir)
    chosen_accounts = save_example_trajectory_plot(panel, args.figures_dir)
    summary = summarize_transition_analysis(panel, counts, probabilities, chosen_accounts)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    (args.tables_dir / "transition_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
