import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="ML Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOG_PATHS = {
    "drift": "logs/drift_log.json",
    "metrics": "logs/metrics_log.json",
    "validation": "logs/validation_log.json",
    "alerts": "logs/alerts.json",
}

BASELINES = {"auc": 0.65, "ndcg_at_10": 0.28, "precision_at_10": 0.18}
ALERT_THRESHOLDS = {"psi": 0.2, "ks": 0.05}


def load_log(path: str) -> List[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def severity_badge(drifted: bool) -> str:
    return "🔴 Drift" if drifted else "🟢 OK"


# Sidebar
st.sidebar.title("⚙️ ML Monitor")
st.sidebar.markdown("**Model:** XGBoost Recommender")
st.sidebar.markdown("**Dataset:** UCI Online Retail")
st.sidebar.divider()
st.sidebar.markdown("**Thresholds**")
st.sidebar.markdown(f"PSI alert: `{ALERT_THRESHOLDS['psi']}`")
st.sidebar.markdown(f"KS p-value: `{ALERT_THRESHOLDS['ks']}`")
st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# Load logs
drift_logs = load_log(LOG_PATHS["drift"])
metrics_logs = load_log(LOG_PATHS["metrics"])
validation_logs = load_log(LOG_PATHS["validation"])
alert_logs = load_log(LOG_PATHS["alerts"])

# Header
st.title("📊 ML Model Monitoring Dashboard")
st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

if not any([drift_logs, metrics_logs, alert_logs]):
    st.warning(
        "No monitoring logs found. Run "
        "`python run_monitoring.py --batch data/production/batch_003.parquet --with-labels`"
    )

# KPI Row
col1, col2, col3, col4 = st.columns(4)

total_batches = len(drift_logs)
drift_count = sum(1 for d in drift_logs if d.get("overall_drift_detected"))
val_failures = sum(1 for v in validation_logs if not v.get("passed", True))
total_alerts = sum(a.get("total_alerts", 0) for a in alert_logs)

col1.metric("Batches Monitored", total_batches)
col2.metric("Batches with Drift", drift_count, delta_color="inverse")
col3.metric("Validation Failures", val_failures, delta_color="inverse")
col4.metric("Total Alerts Fired", total_alerts, delta_color="inverse")

st.divider()

# Drift Trends
st.subheader("📉 Feature Drift Over Time")

if drift_logs:
    records = []
    for log in drift_logs:
        ts = log.get("timestamp", "")[:19]
        for feat, info in log.get("features", {}).items():
            method = info.get("method")
            score = info.get("psi") if method == "psi" else info.get("ks_statistic")
            records.append({
                "Timestamp": ts,
                "Feature": feat,
                "Score": score,
                "Method": method,
                "Drifted": info.get("drifted", False),
            })

    if records:
        df_drift = pd.DataFrame(records)
        fig = px.line(
            df_drift,
            x="Timestamp",
            y="Score",
            color="Feature",
            markers=True,
            title="PSI / KS Score per Feature (higher = more drift)",
        )
        fig.add_hline(
            y=ALERT_THRESHOLDS["psi"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"PSI threshold ({ALERT_THRESHOLDS['psi']})",
            annotation_position="top left",
        )
        fig.add_hline(
            y=ALERT_THRESHOLDS["ks"],
            line_dash="dot",
            line_color="orange",
            annotation_text=f"KS threshold ({ALERT_THRESHOLDS['ks']})",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Latest batch status table
        latest_drift = drift_logs[-1].get("features", {})
        if latest_drift:
            st.markdown("**Latest batch — per-feature status:**")
            status_rows = []
            for feat, info in latest_drift.items():
                method = info.get("method")
                score = info.get("psi") if method == "psi" else info.get("ks_statistic")
                status_rows.append({
                    "Feature": feat,
                    "Method": method.upper().replace("_", " "),
                    "Score": f"{score:.4f}",
                    "Status": severity_badge(info.get("drifted", False)),
                })
            st.dataframe(
                pd.DataFrame(status_rows),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("No drift logs yet.")

st.divider()

# Performance Metrics
st.subheader("📈 Model Performance Over Time")

if metrics_logs:
    col_left, col_right = st.columns([1, 2])

    latest_metrics = metrics_logs[-1].get("metrics", {})
    with col_left:
        st.markdown("**Latest batch vs baseline:**")
        for name, val in latest_metrics.items():
            base = BASELINES.get(name, 1.0)
            delta = val - base
            st.metric(
                label=name.replace("_", " ").upper(),
                value=f"{val:.4f}",
                delta=f"{delta:+.4f}",
                delta_color="normal",
            )

    with col_right:
        perf_records = []
        for log in metrics_logs:
            ts = log.get("timestamp", "")[:19]
            for metric, value in log.get("metrics", {}).items():
                perf_records.append({
                    "Timestamp": ts,
                    "Metric": metric,
                    "Value": value,
                })

        if perf_records:
            df_perf = pd.DataFrame(perf_records)
            fig2 = px.line(
                df_perf,
                x="Timestamp",
                y="Value",
                color="Metric",
                markers=True,
                title="Metrics Trend",
            )
            for name, base_val in BASELINES.items():
                fig2.add_hline(
                    y=base_val,
                    line_dash="dot",
                    opacity=0.4,
                    annotation_text=f"{name} baseline",
                )
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No performance logs yet. Run with `--with-labels` flag.")

st.divider()

# Recent Alerts
st.subheader("🚨 Recent Alerts")

if alert_logs:
    fired_alerts = [a for a in alert_logs if a.get("total_alerts", 0) > 0]
    if fired_alerts:
        for entry in reversed(fired_alerts[-10:]):
            with st.expander(
                f"🔴  {entry['timestamp'][:19]}  —  {entry['total_alerts']} alert(s)"
            ):
                for alert in entry.get("alerts", []):
                    st.markdown(f"**Type:** `{alert['type']}` | **Severity:** `{alert['severity']}`")
                    st.markdown(f"**Details:** {alert['details']}")
                    st.divider()
    else:
        st.success("✅ No alerts fired across all monitored batches.")
else:
    st.info("No alert logs found.")

st.caption(
    "ML Monitoring Framework | XGBoost Recommender | UCI Online Retail | "
    "github.com/Sonal12061/ml-monitoring-framework"
)