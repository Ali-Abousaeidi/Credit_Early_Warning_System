"""Streamlit dashboard for the credit early-warning system.

Run after generating artifacts with:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict:
    """Load a JSON report file."""
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_table(path: str) -> pd.DataFrame:
    """Load a CSV report table."""
    return pd.read_csv(ROOT / path)


st.set_page_config(page_title="Credit Early-Warning System", layout="wide")
st.title("Credit Early-Warning System")

watchlist_summary = load_json(REPORTS / "tables" / "watchlist_summary.json")
model_metrics = load_json(REPORTS / "tables" / "model_metrics.json")
audit = load_json(REPORTS / "tables" / "methodology_audit.json")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Median lead time", f"{watchlist_summary['median_lead_time_months']:.1f} months")
col2.metric("Precision@100", f"{watchlist_summary['precision_at_k_overall']:.1%}")
col3.metric("Account capture", f"{watchlist_summary['account_capture_rate']:.1%}")
col4.metric("Audit", audit["overall_status"])

st.subheader("Out-of-Time Model Metrics")
st.json(model_metrics["test_metrics"])

left, right = st.columns(2)
with left:
    st.image(str(REPORTS / "figures" / "lead_time.png"), caption="Lead-time distribution")
    st.image(
        str(REPORTS / "figures" / "capacity_sensitivity.png"),
        caption="Capacity sensitivity",
    )
with right:
    st.image(
        str(REPORTS / "figures" / "model_precision_recall.png"),
        caption="Precision-recall curve",
    )
    st.image(
        str(REPORTS / "figures" / "reason_feature_frequency.png"),
        caption="Reason-code frequency",
    )

st.subheader("Watchlist")
watchlist = load_table("reports/tables/watchlist_reason_codes.csv")
month_options = sorted(watchlist["observation_month"].unique())
selected_month = st.selectbox("Observation month", month_options)
month_watchlist = watchlist.loc[watchlist["observation_month"].eq(selected_month)]
st.dataframe(
    month_watchlist[
        [
            "account_id",
            "watchlist_rank",
            "predicted_probability",
            "target_deterioration_6m",
            "months_to_deterioration",
            "reason_1",
            "reason_2",
            "reason_3",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)
