import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order[:k]]

    gains = 2.0 ** y_sorted - 1.0
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains / discounts))

    ideal = np.sort(y_true)[::-1][:k]
    ideal_gains = 2.0 ** ideal - 1.0
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float(np.sum(ideal_gains / ideal_discounts))

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(y_score)[::-1]
    return float(np.mean(y_true[order[:k]]))


class PerformanceTracker:

    def __init__(self, config: Dict[str, Any], log_path: str = "logs/metrics_log.json"):
        self.config = config
        self.log_path = log_path
        perf_cfg = config.get("monitoring", {}).get("performance", {})

        self.baseline = {
            "auc": perf_cfg.get("baseline_auc", 0.9939),
            "ndcg_at_10": perf_cfg.get("baseline_ndcg", 0.9689),
            "precision_at_10": perf_cfg.get("baseline_precision", 0.8729),
        }
        self.thresholds = {
            "auc": perf_cfg.get("auc_drop_threshold", 0.05),
            "ndcg_at_10": perf_cfg.get("ndcg_drop_threshold", 0.05),
            "precision_at_10": perf_cfg.get("precision_drop_threshold", 0.05),
        }

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        batch_timestamp: Optional[str] = None,
        k: int = 10,
    ) -> Dict[str, Any]:

        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class in y_true — AUC undefined.")
            return {"error": "insufficient_classes", "timestamp": datetime.utcnow().isoformat()}

        auc = float(roc_auc_score(y_true, y_score))
        ndcg = ndcg_at_k(y_true, y_score, k)
        prec = precision_at_k(y_true, y_score, k)

        current_metrics = {
            "auc": round(auc, 6),
            "ndcg_at_10": round(ndcg, 6),
            "precision_at_10": round(prec, 6),
        }

        degradation = {}
        for name, value in current_metrics.items():
            drop = self.baseline[name] - value
            degradation[name] = {
                "current": value,
                "baseline": self.baseline[name],
                "drop": round(drop, 6),
                "threshold": self.thresholds[name],
                "alert": drop > self.thresholds[name],
            }

        any_alert = any(v["alert"] for v in degradation.values())

        report = {
            "timestamp": batch_timestamp or datetime.utcnow().isoformat(),
            "logged_at": datetime.utcnow().isoformat(),
            "sample_size": len(y_true),
            "metrics": current_metrics,
            "degradation_check": degradation,
            "performance_alert": any_alert,
        }

        self._log_result(report)

        if any_alert:
            degraded_names = [k for k, v in degradation.items() if v["alert"]]
            logger.warning(f"[PERFORMANCE ALERT] Degradation in: {degraded_names}")
        else:
            logger.info(
                f"Performance OK | AUC={auc:.4f} | NDCG@10={ndcg:.4f} | P@10={prec:.4f}"
            )

        return report

    def get_history(self, last_n: int = 50) -> List[Dict]:
        try:
            with open(self.log_path) as f:
                logs = json.load(f)
            return logs[-last_n:]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

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
            logger.error(f"Failed to write metrics log: {e}")