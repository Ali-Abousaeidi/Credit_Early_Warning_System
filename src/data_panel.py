"""Build the account-month monitoring panel.

The project uses synthetic data because public credit datasets rarely expose a
clean account-month behavioural panel. The synthetic process is deliberately
simple enough to audit: accounts develop latent stress, some enter a
pre-delinquency drift phase, and delinquency status is recorded at each monthly
snapshot.

Important timing convention: this module writes only information known at the
monthly observation point. It does not expose the internally simulated future
deterioration month; labels are created later by looking forward from the
written status history.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src import RANDOM_SEED

STATUS_DPD: Final[dict[str, int]] = {
    "current": 0,
    "dpd_30": 30,
    "dpd_60": 60,
    "dpd_90": 90,
    "default": 120,
}


@dataclass(frozen=True)
class PanelConfig:
    """Configuration for the reproducible synthetic monitoring panel."""

    n_accounts: int = 2_500
    n_months: int = 36
    start_month: str = "2020-01-31"
    seed: int = RANDOM_SEED


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Logistic transform for risk probabilities."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _calendar(config: PanelConfig) -> pd.DatetimeIndex:
    """Return month-end observation dates."""
    return pd.date_range(config.start_month, periods=config.n_months, freq="M")


def _simulate_account_static(
    rng: np.random.Generator, config: PanelConfig
) -> pd.DataFrame:
    """Create static account attributes used by the behavioural simulator."""
    account_idx = np.arange(config.n_accounts)
    latent_credit_risk = rng.normal(0.0, 1.0, config.n_accounts)
    credit_score = np.clip(
        705.0 - 65.0 * latent_credit_risk + rng.normal(0.0, 28.0, config.n_accounts),
        520.0,
        840.0,
    ).round()
    origination_pd = _sigmoid(-3.0 + 0.95 * latent_credit_risk)
    monthly_income = np.clip(
        rng.lognormal(mean=np.log(4_800), sigma=0.35, size=config.n_accounts),
        1_800,
        18_000,
    )
    debt_to_income = np.clip(
        0.22 + 0.08 * latent_credit_risk + rng.normal(0.0, 0.06, config.n_accounts),
        0.05,
        0.62,
    )
    credit_limit = np.clip(
        monthly_income
        * rng.uniform(1.8, 4.0, config.n_accounts)
        * (1.10 - 0.45 * origination_pd),
        1_500,
        60_000,
    ).round(2)
    starting_utilization = np.clip(
        0.28 + 0.13 * latent_credit_risk + rng.normal(0.0, 0.10, config.n_accounts),
        0.03,
        0.88,
    )
    apr = np.clip(0.055 + 0.28 * origination_pd, 0.055, 0.295)
    origination_offsets = rng.integers(-24, 1, config.n_accounts)
    observed_start = _calendar(config)[0]
    origination_month = [
        observed_start + pd.offsets.MonthEnd(int(offset))
        for offset in origination_offsets
    ]

    return pd.DataFrame(
        {
            "account_id": [f"ACC{idx:06d}" for idx in account_idx],
            "origination_month": origination_month,
            "credit_score_at_origination": credit_score.astype(int),
            "origination_pd": origination_pd.round(5),
            "monthly_income": monthly_income.round(2),
            "debt_to_income": debt_to_income.round(4),
            "credit_limit": credit_limit,
            "apr": apr.round(5),
            "starting_utilization": starting_utilization.round(4),
            "latent_credit_risk": latent_credit_risk,
        }
    )


def _sample_event_months(
    rng: np.random.Generator, static: pd.DataFrame, config: PanelConfig
) -> np.ndarray:
    """Sample hidden deterioration months used only inside the simulator."""
    event_probability = _sigmoid(-1.75 + 0.95 * static["latent_credit_risk"].to_numpy())
    ever_deteriorates = rng.random(config.n_accounts) < event_probability
    event_month = np.full(config.n_accounts, fill_value=-1, dtype=int)

    eligible_idx = np.flatnonzero(ever_deteriorates)
    if len(eligible_idx) == 0:
        return event_month

    min_event_month = min(8, config.n_months - 1)
    possible_months = np.arange(min_event_month, config.n_months)
    macro_weights = np.linspace(0.75, 1.35, len(possible_months))
    macro_weights += 0.35 * np.exp(-0.5 * ((possible_months - 25) / 4.0) ** 2)
    macro_weights = macro_weights / macro_weights.sum()
    event_month[eligible_idx] = rng.choice(
        possible_months, size=len(eligible_idx), replace=True, p=macro_weights
    )
    return event_month


def _status_for_month(rng: np.random.Generator, months_since_event: int) -> str:
    """Return delinquency status after the first 30 DPD event."""
    if months_since_event < 0:
        return "current"
    if months_since_event == 0:
        return "dpd_30"
    if months_since_event == 1:
        return str(rng.choice(["current", "dpd_30", "dpd_60"], p=[0.18, 0.25, 0.57]))
    if months_since_event == 2:
        return str(
            rng.choice(
                ["current", "dpd_30", "dpd_60", "dpd_90"],
                p=[0.18, 0.12, 0.25, 0.45],
            )
        )
    if months_since_event == 3:
        return str(
            rng.choice(
                ["current", "dpd_60", "dpd_90", "default"],
                p=[0.15, 0.15, 0.30, 0.40],
            )
        )
    return str(rng.choice(["current", "dpd_90", "default"], p=[0.12, 0.18, 0.70]))


def build_synthetic_panel(config: PanelConfig = PanelConfig()) -> pd.DataFrame:
    """Build an account-month panel with realistic deterioration trajectories.

    The returned panel includes current behavioural state and delinquency status
    only. It excludes forward target labels and hidden simulation event dates.
    """
    rng = np.random.default_rng(config.seed)
    dates = _calendar(config)
    static = _simulate_account_static(rng, config)
    event_months = _sample_event_months(rng, static, config)

    rows: list[dict[str, object]] = []
    month_axis = np.arange(config.n_months)
    macro_stress = 0.18 * np.sin(np.linspace(-0.4, 2.8 * np.pi, config.n_months))
    macro_stress += 0.45 * np.exp(-0.5 * ((month_axis - 25) / 4.0) ** 2)

    for account_pos, account in static.iterrows():
        latent_risk = float(account["latent_credit_risk"])
        credit_limit = float(account["credit_limit"])
        previous_utilization = float(account["starting_utilization"])
        account_stress = latent_risk + rng.normal(0.0, 0.25)
        default_absorbed = False

        for month_idx, observation_month in enumerate(dates):
            hidden_event_month = event_months[account_pos]
            months_to_event = (
                hidden_event_month - month_idx if hidden_event_month >= 0 else 999
            )
            months_since_event = (
                month_idx - hidden_event_month if hidden_event_month >= 0 else -1
            )
            pre_delinquency_pressure = (
                (7 - months_to_event) / 6.0 if 1 <= months_to_event <= 6 else 0.0
            )
            account_stress = (
                0.72 * account_stress
                + 0.18 * latent_risk
                + macro_stress[month_idx]
                + 0.62 * pre_delinquency_pressure
                + rng.normal(0.0, 0.22)
            )

            if default_absorbed:
                status = "default"
            else:
                status = _status_for_month(rng, months_since_event)
                default_absorbed = status == "default"

            utilization = np.clip(
                0.78 * previous_utilization
                + 0.08
                + 0.045 * latent_risk
                + 0.19 * pre_delinquency_pressure
                + 0.035 * account_stress
                + rng.normal(0.0, 0.035),
                0.01,
                1.18,
            )
            if status in {"dpd_60", "dpd_90", "default"}:
                utilization = min(1.25, utilization + 0.06)

            balance = max(0.0, credit_limit * utilization)
            minimum_payment_due = max(25.0, balance * (0.022 + float(account["apr"]) / 12))
            scheduled_payment_due = max(minimum_payment_due, balance * 0.045)
            payment_ratio = np.clip(
                1.08
                - 0.36 * pre_delinquency_pressure
                - 0.10 * account_stress
                - (0.20 if status != "current" else 0.0)
                + rng.normal(0.0, 0.12),
                0.0,
                1.55,
            )
            actual_payment = scheduled_payment_due * payment_ratio
            cash_buffer_months = np.clip(
                3.6
                - 1.25 * pre_delinquency_pressure
                - 0.33 * account_stress
                + rng.normal(0.0, 0.45),
                0.0,
                8.0,
            )
            purchase_amount = np.clip(
                float(account["monthly_income"])
                * (0.20 + 0.05 * latent_risk + rng.normal(0.0, 0.05)),
                0.0,
                credit_limit * 0.45,
            )
            days_past_due = STATUS_DPD[status]
            account_age_months = (
                observation_month.year - account["origination_month"].year
            ) * 12 + (observation_month.month - account["origination_month"].month)

            rows.append(
                {
                    "account_id": account["account_id"],
                    "observation_month": observation_month,
                    "month_index": month_idx,
                    "account_age_months": int(account_age_months),
                    "origination_month": account["origination_month"],
                    "credit_score_at_origination": int(
                        account["credit_score_at_origination"]
                    ),
                    "origination_pd": float(account["origination_pd"]),
                    "monthly_income": round(float(account["monthly_income"]), 2),
                    "debt_to_income": round(float(account["debt_to_income"]), 4),
                    "credit_limit": round(credit_limit, 2),
                    "apr": round(float(account["apr"]), 5),
                    "balance": round(balance, 2),
                    "utilization": round(float(utilization), 4),
                    "minimum_payment_due": round(minimum_payment_due, 2),
                    "scheduled_payment_due": round(scheduled_payment_due, 2),
                    "actual_payment": round(actual_payment, 2),
                    "payment_to_due_ratio": round(float(payment_ratio), 4),
                    "purchase_amount": round(float(purchase_amount), 2),
                    "cash_buffer_months": round(float(cash_buffer_months), 3),
                    "missed_min_payment": bool(payment_ratio < 0.82 or status != "current"),
                    "days_past_due": days_past_due,
                    "performance_status": status,
                    "macro_stress_index": round(float(macro_stress[month_idx]), 4),
                }
            )
            previous_utilization = float(utilization)

    panel = pd.DataFrame(rows)
    return panel.sort_values(["account_id", "observation_month"]).reset_index(drop=True)


def summarize_panel(panel: pd.DataFrame, config: PanelConfig) -> dict[str, object]:
    """Create reproducibility and quality-check metadata for a generated panel."""
    status_counts = panel["performance_status"].value_counts().sort_index()
    terminal_status = panel.groupby("account_id")["performance_status"].agg(
        lambda values: values.iloc[-1]
    )
    deteriorated_accounts = panel.loc[panel["days_past_due"] >= 30, "account_id"].nunique()
    return {
        "config": asdict(config),
        "n_rows": int(len(panel)),
        "n_accounts": int(panel["account_id"].nunique()),
        "months_per_account_min": int(panel.groupby("account_id").size().min()),
        "months_per_account_max": int(panel.groupby("account_id").size().max()),
        "observation_month_min": str(panel["observation_month"].min().date()),
        "observation_month_max": str(panel["observation_month"].max().date()),
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "accounts_ever_30_plus_dpd": int(deteriorated_accounts),
        "accounts_ever_30_plus_dpd_rate": round(
            deteriorated_accounts / config.n_accounts, 4
        ),
        "terminal_status_counts": {
            key: int(value) for key, value in terminal_status.value_counts().sort_index().items()
        },
    }


def write_panel(
    panel: pd.DataFrame, output_path: Path, metadata_path: Path, config: PanelConfig
) -> dict[str, object]:
    """Write the panel CSV and metadata JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    metadata = summarize_panel(panel, config)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    """Build and write the synthetic monitoring panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-accounts", type=int, default=PanelConfig.n_accounts)
    parser.add_argument("--n-months", type=int, default=PanelConfig.n_months)
    parser.add_argument("--start-month", type=str, default=PanelConfig.start_month)
    parser.add_argument("--seed", type=int, default=PanelConfig.seed)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/panel/account_month_panel.csv"),
        help="Path for the generated account-month panel CSV.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/panel/panel_metadata.json"),
        help="Path for panel metadata and quality-check summary.",
    )
    args = parser.parse_args()

    config = PanelConfig(
        n_accounts=args.n_accounts,
        n_months=args.n_months,
        start_month=args.start_month,
        seed=args.seed,
    )
    panel = build_synthetic_panel(config)
    metadata = write_panel(panel, args.output, args.metadata, config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
