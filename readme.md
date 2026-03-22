# ML Monitoring and Observability Framework

A production-grade ML monitoring framework built on top of an XGBoost 
purchase-prediction model (UCI Online Retail dataset). Implements automated 
data validation, feature and prediction drift detection, model performance 
tracking with degradation alerting, and CI/CD-triggered retraining.

---

## Architecture
```
Production Data Batch
        │
        ├──► Data Validator      (pandera schema + null checks)
        ├──► Drift Detector      (PSI for categorical, KS test for continuous)
        └──► Performance Tracker (AUC-ROC, NDCG@10, Precision@10)
                │
                ▼
          Alert Engine           (email / Slack webhook)
                │
        ┌───────┴──────────┐
        ▼                  ▼
  Streamlit           GitHub Actions
  Dashboard           workflow_dispatch
                           │
                           ▼
                    Retrain Pipeline
```

---

## Components

| Component | File | What it does |
|---|---|---|
| Data Validator | `monitoring/data_validator.py` | pandera schema, null rate checks, outlier flags |
| Drift Detector | `monitoring/drift_detector.py` | PSI (categorical), KS test (continuous), prediction drift |
| Performance Tracker | `monitoring/performance_tracker.py` | AUC, NDCG@10, P@10 vs baseline; delayed-label support |
| Alert Engine | `monitoring/alerting.py` | Threshold evaluation, email (SMTP), Slack webhook |
| Retraining Trigger | `retraining/trigger.py` | Consecutive drift counter → GitHub Actions dispatch |
| Dashboard | `dashboard/app.py` | Streamlit live view of all logs |
| CI Monitor | `.github/workflows/monitor.yml` | Scheduled 6-hourly monitoring run |
| CI Retrain | `.github/workflows/retrain.yml` | Triggered by drift; retrains and commits model |

---

## Drift Detection — Method Reference

### PSI (Population Stability Index) — categorical features
```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```
| PSI value | Interpretation |
|---|---|
| < 0.1 | No significant drift |
| 0.1 – 0.2 | Moderate drift — monitor |
| > 0.2 | Significant drift — alert + retrain |

### KS Test (Kolmogorov-Smirnov) — continuous features
Measures maximum vertical distance between reference and production CDFs.
`p-value < 0.05` → distributions are statistically different → drift alert.

---

## Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/Sonal12061/ml-monitoring-framework
cd ml-monitoring-framework

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate reference data + 6 simulated production batches
python generate_data.py

# 5. Run a clean batch (no drift expected)
python run_monitoring.py --batch data/production/batch_000.parquet --with-labels

# 6. Run a drifted batch (alerts expected)
python run_monitoring.py --batch data/production/batch_004.parquet --with-labels

# 7. Launch the monitoring dashboard
streamlit run dashboard/app.py
```

---

## Simulated Drift Levels

| Batch | Drift factor | Expected behaviour |
|---|---|---|
| `batch_000` | 0.00 | All checks pass |
| `batch_001` | 0.05 | All checks pass |
| `batch_002` | 0.15 | Approaching threshold |
| `batch_003` | 0.35 | KS / PSI alerts fire |
| `batch_004` | 0.60 | Consecutive counter → retraining dispatched |
| `batch_005` | 0.80 | All systems alarming |

---

## Configuration

All thresholds live in `config.yaml`:
```yaml
monitoring:
  drift:
    psi_threshold: 0.2      # PSI above this → alert
    ks_threshold: 0.05      # p-value below this → alert
  performance:
    auc_drop_threshold: 0.05
    baseline_auc: 0.65
retraining:
  drift_trigger_count: 2    # trigger after N consecutive drifted batches
```

---

## CI/CD Behavior

The monitoring workflow exits with code 1 when alerts are fired — 
this is intentional. A red X in GitHub Actions means drift or 
degradation was detected, not that the pipeline broke.

| Exit code | Meaning |
|---|---|
| 0 | All checks passed — no drift, no alerts |
| 1 | Alerts fired — drift or performance degradation detected |

Artifacts (monitoring logs) are uploaded regardless of exit code 
and can be downloaded from the workflow run summary page.

> In production, logs would be persisted to S3 or a database so 
> the Streamlit dashboard retains historical trends across runs 
> rather than resetting each CI cycle.

## GitHub Actions Setup

1. Add `GITHUB_TOKEN` as a repo secret (Settings → Secrets → Actions)
2. Monitor workflow runs every 6 hours automatically
3. When drift persists for 2+ consecutive batches → `retrain.yml` triggers
   → model retrains → committed back to repo

---

## Baseline Model Performance

| Metric | Baseline |
|---|---|
| AUC-ROC | 0.65 |
| NDCG@10 | 0.28 |
| Precision@10 | 0.18 |

---

## Tech Stack

`Python` · `XGBoost` · `pandera` · `scipy` · `scikit-learn` · 
`Streamlit` · `Plotly` · `GitHub Actions`

---

## Related Projects

- [E-commerce Recommendation System](https://github.com/Sonal12061/ecommerce-recommendation-xgboost) — the XGBoost model being monitored
