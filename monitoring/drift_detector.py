import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:

    PSI_BINS = 10
    PSI_EPSILON = 1e-6

    def __init__(
        self,
        reference_df: pd.DataFrame,
        config: Dict[str, Any],
        log_path: str = "logs/drift_log.json",
    ):
        self.reference_df = reference_df
        self.config = config
        self.log_path = log_path

        drift_cfg = config.get("monitoring", {}).get("drift", {})
        self.psi_threshold = drift_cfg.get("psi_threshold", 0.2)
        self.ks_threshold = drift_cfg.get("ks_threshold", 0.05)

        features = config.get("model", {}).get("features", {})
        self.numerical_features: List[str] = features.get("numerical", [])
        self.categorical_features: List[str] = features.get("categorical", [])

    def compute_psi(self, reference: np.ndarray, production: np.ndarray) -> float:
        breakpoints = np.percentile(reference, np.linspace(0, 100, self.PSI_BINS + 1))
        breakpoints = np.unique(breakpoints)

        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        prod_counts, _ = np.histogram(production, bins=breakpoints)

        ref_pct = ref_counts / len(reference) + self.PSI_EPSILON
        prod_pct = prod_counts / len(production) + self.PSI_EPSILON

        psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return psi

    def compute_ks(self, reference: np.ndarray, production: np.ndarray) -> Dict[str, float]:
        stat, pvalue = stats.ks_2samp(reference, production)
        return {"ks_statistic": float(stat), "p_value": float(pvalue)}

    def detect(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "batch_size": len(production_df),
            "overall_drift_detected": False,
            "features": {},
            "prediction_drift": None,
        }

        # Numerical features: KS test
        for feature in self.numerical_features:
            if feature not in self.reference_df.columns or feature not in production_df.columns:
                logger.warning(f"Feature '{feature}' missing — skipping.")
                continue

            ref_vals = self.reference_df[feature].dropna().values
            prod_vals = production_df[feature].dropna().values
            ks = self.compute_ks(ref_vals, prod_vals)
            drifted = ks["p_value"] < self.ks_threshold

            report["features"][feature] = {
                "method": "ks_test",
                "ks_statistic": round(ks["ks_statistic"], 6),
                "p_value": round(ks["p_value"], 6),
                "threshold": self.ks_threshold,
                "drifted": drifted,
            }
            if drifted:
                report["overall_drift_detected"] = True
                logger.warning(
                    f"[DRIFT] {feature}: KS={ks['ks_statistic']:.4f}, p={ks['p_value']:.4f}"
                )

        # Categorical features: PSI
        for feature in self.categorical_features:
            if feature not in self.reference_df.columns or feature not in production_df.columns:
                continue

            ref_vals = self.reference_df[feature].dropna().values.astype(float)
            prod_vals = production_df[feature].dropna().values.astype(float)
            psi = self.compute_psi(ref_vals, prod_vals)
            drifted = psi > self.psi_threshold

            report["features"][feature] = {
                "method": "psi",
                "psi": round(psi, 6),
                "threshold": self.psi_threshold,
                "severity": "high" if psi > 0.2 else ("moderate" if psi > 0.1 else "low"),
                "drifted": drifted,
            }
            if drifted:
                report["overall_drift_detected"] = True
                logger.warning(f"[DRIFT] {feature}: PSI={psi:.4f}")

        # Prediction drift: KS test
        if "prediction" in production_df.columns and "prediction" in self.reference_df.columns:
            ref_preds = self.reference_df["prediction"].dropna().values
            prod_preds = production_df["prediction"].dropna().values
            ks = self.compute_ks(ref_preds, prod_preds)
            drifted = ks["p_value"] < self.ks_threshold
            if drifted:
                report["overall_drift_detected"] = True
            report["prediction_drift"] = {
                "method": "ks_test",
                "ks_statistic": round(ks["ks_statistic"], 6),
                "p_value": round(ks["p_value"], 6),
                "drifted": drifted,
            }

        self._log_result(report)
        status = "DETECTED" if report["overall_drift_detected"] else "NONE"
        logger.info(f"Drift check complete | Status: {status}")
        return report

    def _log_result(self, report: Dict[str, Any]) -> None:
        try:
            try:
                with open(self.log_path) as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
            logs.append(report)
            with open(self.log_path, "w") as f:
                json.dump(logs, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write drift log: {e}")