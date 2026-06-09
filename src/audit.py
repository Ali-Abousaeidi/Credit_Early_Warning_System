"""Methodology and reproducibility audit checks.

The audit is deliberately lightweight and transparent. It checks key
methodological risks: hidden future columns, feature/target leakage, time-split
ordering, complete target windows, and watchlist capacity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.features import feature_columns
from src.model import TARGET_COLUMN

FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "future",
    "first_deterioration",
    "months_to_deterioration",
    "label",
)


def _pass(message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a passing audit item."""
    return {"status": "pass", "message": message, "details": details or {}}


def _fail(message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a failing audit item."""
    return {"status": "fail", "message": message, "details": details or {}}


def load_csv(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV if it exists, parsing provided date columns when possible."""
    if not path.exists():
        raise FileNotFoundError(f"Required audit input does not exist: {path}")
    return pd.read_csv(path, parse_dates=date_columns or [])


def audit_raw_panel(panel: pd.DataFrame) -> dict[str, Any]:
    """Check the raw panel has no target/future leakage columns."""
    forbidden_columns = [
        column
        for column in panel.columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden_columns:
        return _fail(
            "Raw panel contains future/target-like columns.",
            {"columns": forbidden_columns},
        )
    duplicates = int(panel[["account_id", "observation_month"]].duplicated().sum())
    if duplicates:
        return _fail("Raw panel has duplicate account-month rows.", {"duplicates": duplicates})
    return _pass(
        "Raw panel contains one row per account-month and no future/target columns.",
        {
            "rows": int(len(panel)),
            "accounts": int(panel["account_id"].nunique()),
        },
    )


def audit_feature_columns(feature_matrix: pd.DataFrame) -> dict[str, Any]:
    """Check selected model columns exclude target/timing metadata."""
    columns = feature_columns(feature_matrix)
    forbidden_columns = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden_columns:
        return _fail(
            "Model feature list contains forbidden future/target-like columns.",
            {"columns": forbidden_columns},
        )
    return _pass(
        "Model feature list excludes forward target and timing metadata.",
        {"n_features": len(columns)},
    )


def audit_time_split(scored: pd.DataFrame) -> dict[str, Any]:
    """Check train/calibration/test windows are strictly ordered."""
    ranges = (
        scored.groupby("split")["observation_month"]
        .agg(["min", "max"])
        .to_dict(orient="index")
    )
    required = {"train", "calibration", "test"}
    missing = required.difference(ranges)
    if missing:
        return _fail("Missing required split(s).", {"missing": sorted(missing)})
    ordered = (
        ranges["train"]["max"] < ranges["calibration"]["min"]
        and ranges["calibration"]["max"] < ranges["test"]["min"]
    )
    if not ordered:
        return _fail("Time splits are not strictly ordered.", {"ranges": str(ranges)})
    return _pass(
        "Train, calibration, and test windows are strictly ordered by time.",
        {
            split: {
                "min": str(values["min"].date()),
                "max": str(values["max"].date()),
            }
            for split, values in ranges.items()
        },
    )


def audit_target_window(target_population: pd.DataFrame) -> dict[str, Any]:
    """Check labelled rows are current and have a complete outcome window."""
    non_current = int((~target_population["eligible_at_observation"].astype(bool)).sum())
    incomplete = int((~target_population["has_full_outcome_window"].astype(bool)).sum())
    if non_current or incomplete:
        return _fail(
            "Target population contains ineligible rows.",
            {"non_current": non_current, "incomplete_window": incomplete},
        )
    return _pass(
        "Target population uses current rows with complete forward outcome windows.",
        {
            "rows": int(len(target_population)),
            "positive_rate": round(float(target_population[TARGET_COLUMN].mean()), 4),
        },
    )


def audit_watchlist_capacity(watchlist: pd.DataFrame, top_k_per_month: int) -> dict[str, Any]:
    """Check the watchlist respects the stated monthly review capacity."""
    monthly_counts = watchlist.groupby("observation_month").size()
    if monthly_counts.gt(top_k_per_month).any():
        return _fail(
            "Watchlist exceeds stated monthly capacity.",
            {
                str(month.date()): int(count)
                for month, count in monthly_counts.loc[monthly_counts.gt(top_k_per_month)].items()
            },
        )
    return _pass(
        "Watchlist respects monthly top-k review capacity.",
        {
            "top_k_per_month": top_k_per_month,
            "months": int(monthly_counts.shape[0]),
            "rows": int(len(watchlist)),
        },
    )


def run_audit(
    panel_path: Path,
    target_path: Path,
    feature_path: Path,
    scored_path: Path,
    watchlist_path: Path,
    top_k_per_month: int = 100,
) -> dict[str, Any]:
    """Run all audit checks and return a serializable report."""
    panel = load_csv(panel_path, ["observation_month", "origination_month"])
    target_population = load_csv(
        target_path,
        ["observation_month", "origination_month", "first_deterioration_month"],
    )
    feature_matrix = load_csv(
        feature_path,
        ["observation_month", "origination_month", "first_deterioration_month"],
    )
    scored = load_csv(
        scored_path,
        ["observation_month", "origination_month", "first_deterioration_month"],
    )
    watchlist = load_csv(
        watchlist_path,
        ["observation_month", "origination_month", "first_deterioration_month"],
    )

    checks = {
        "raw_panel": audit_raw_panel(panel),
        "feature_columns": audit_feature_columns(feature_matrix),
        "time_split": audit_time_split(scored),
        "target_window": audit_target_window(target_population),
        "watchlist_capacity": audit_watchlist_capacity(watchlist, top_k_per_month),
    }
    failed = {name: check for name, check in checks.items() if check["status"] != "pass"}
    return {
        "overall_status": "pass" if not failed else "fail",
        "checks": checks,
    }


def main() -> None:
    """Run audit checks against generated pipeline artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/panel/account_month_panel.csv"))
    parser.add_argument("--target", type=Path, default=Path("data/targets/target_population.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/feature_matrix.csv"))
    parser.add_argument("--scored", type=Path, default=Path("data/model/scored_feature_matrix.csv"))
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=Path("reports/tables/watchlist_top100_test.csv"),
    )
    parser.add_argument("--top-k-per-month", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/methodology_audit.json"),
    )
    args = parser.parse_args()

    report = run_audit(
        panel_path=args.panel,
        target_path=args.target,
        feature_path=args.features,
        scored_path=args.scored,
        watchlist_path=args.watchlist,
        top_k_per_month=args.top_k_per_month,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["overall_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
