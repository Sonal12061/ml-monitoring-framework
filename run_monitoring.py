import argparse
import json
import logging
import os
import sys

import pandas as pd
import yaml

from monitoring import AlertEngine, DataValidator, DriftDetector, PerformanceTracker
from retraining.trigger import RetrainingTrigger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_monitoring")

os.makedirs("logs", exist_ok=True)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_batch(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("  MONITORING RUN SUMMARY")
    print("=" * 55)
    for key, value in summary.items():
        print(f"  {key:<30} {value}")
    print("=" * 55 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ML monitoring pipeline")
    parser.add_argument("--batch", required=True, help="Path to production batch")
    parser.add_argument("--with-labels", action="store_true",
                        help="Batch contains label column for performance tracking")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    logger.info(f"Starting monitoring pipeline | batch={args.batch}")

    config = load_config(args.config)

    reference_df = pd.read_parquet(config["data"]["reference_path"])
    production_df = load_batch(args.batch)
    feature_cols = (
        config["model"]["features"]["numerical"]
        + config["model"]["features"]["categorical"]
    )

    logger.info(f"Reference: {len(reference_df):,} rows | Production: {len(production_df):,} rows")

    # Step 1: Data Validation
    logger.info("─" * 50)
    logger.info("STEP 1: Data Validation")
    validator = DataValidator(config)
    validation_report = validator.validate(production_df[feature_cols])

    # Step 2: Drift Detection
    logger.info("─" * 50)
    logger.info("STEP 2: Drift Detection")
    detector = DriftDetector(reference_df, config)
    drift_report = detector.detect(production_df)

    for feat, info in drift_report["features"].items():
        if info.get("method") == "ks_test":
            score_str = f"KS={info['ks_statistic']:.4f}, p={info['p_value']:.4f}"
        else:
            score_str = f"PSI={info['psi']:.4f}"
        flag = "DRIFT" if info["drifted"] else "ok"
        logger.info(f"  [{flag}] {feat}: {score_str}")

    # Step 3: Performance Tracking
    performance_report = None
    if args.with_labels:
        if "label" not in production_df.columns or "prediction" not in production_df.columns:
            logger.warning("label/prediction columns not found. Skipping.")
        else:
            logger.info("─" * 50)
            logger.info("STEP 3: Performance Tracking")
            tracker = PerformanceTracker(config)
            performance_report = tracker.compute_metrics(
                y_true=production_df["label"].values,
                y_score=production_df["prediction"].values,
            )
            # Add this temporarily
          
    else:
        logger.info("Skipping performance tracking. Use --with-labels to enable.")

    # Step 4: Alert Engine
    logger.info("─" * 50)
    logger.info("STEP 4: Alert Engine")
    alert_engine = AlertEngine(config)
    alert_summary = alert_engine.evaluate_and_alert(
        validation_report=validation_report,
        drift_report=drift_report,
        performance_report=performance_report,
    )

    # Step 5: Retraining Trigger
    logger.info("─" * 50)
    logger.info("STEP 5: Retraining Trigger")
    trigger = RetrainingTrigger(config)
    triggered = trigger.evaluate(alert_summary)

    if triggered:
        logger.warning("RETRAINING WORKFLOW DISPATCHED via GitHub Actions")
    else:
        logger.info("No retraining triggered.")

    # Final Summary
    final = {
        "batch": args.batch,
        "validation_passed": validation_report["passed"],
        "drift_detected": drift_report["overall_drift_detected"],
        "performance_alert": (
            performance_report.get("performance_alert")
            if performance_report else "N/A (no labels)"
        ),
        "total_alerts": alert_summary["total_alerts"],
        "retraining_triggered": triggered,
    }
    print_summary(final)

    return 1 if alert_summary["total_alerts"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())