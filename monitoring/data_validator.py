import json
import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import pandera.pandas as pa

logger = logging.getLogger(__name__)


class DataValidator:

    def __init__(self, config: Dict[str, Any], log_path: str = "logs/validation_log.json"):
        self.config = config
        self.log_path = log_path
        val_cfg = config.get("monitoring", {}).get("validation", {})
        self.null_threshold = val_cfg.get("null_threshold", 0.01)
        self.fail_on_error = val_cfg.get("fail_on_schema_error", True)
        self.schema = self._build_schema()

    def _build_schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            columns={
                "Recency": pa.Column(float, pa.Check.ge(0), nullable=False),
                "Frequency": pa.Column(float, pa.Check.ge(0), nullable=False),
                "Monetary": pa.Column(float, pa.Check.ge(0), nullable=False),
                "AvgOrderValue": pa.Column(float, pa.Check.ge(0), nullable=False),
                "DaysSinceFirst": pa.Column(float, pa.Check.ge(0), nullable=False),
                "Country_encoded": pa.Column(int, pa.Check.isin(range(40)), nullable=False),
            },
            checks=[
                pa.Check(lambda df: df.shape[0] > 0, error="Batch must not be empty"),
            ],
            coerce=True,
        )

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "batch_size": len(df),
            "passed": True,
            "errors": [],
            "warnings": [],
            "null_rates": {},
        }

        # 1. Schema validation
        try:
            self.schema.validate(df, lazy=True)
            logger.info("Schema validation passed.")
        except pa.errors.SchemaErrors as exc:
            errors = exc.failure_cases[["check", "column", "failure_case"]].to_dict("records")
            report["errors"].extend([f"Schema: {e}" for e in errors])
            if self.fail_on_error:
                report["passed"] = False
            logger.warning(f"Schema validation failed: {errors}")

        # 2. Null rate check
        null_rates = df.isnull().mean().to_dict()
        report["null_rates"] = null_rates
        for col, rate in null_rates.items():
            if rate > self.null_threshold:
                msg = f"High null rate in '{col}': {rate:.2%} > threshold {self.null_threshold:.2%}"
                report["errors"].append(msg)
                report["passed"] = False
                logger.warning(msg)

        # 3. Outlier check
        if "Monetary" in df.columns:
            p99 = df["Monetary"].quantile(0.99)
            if p99 > 1_000_000:
                report["warnings"].append(f"Monetary p99={p99:.2f} — possible outlier spike")

        self._log_result(report)
        status = "PASSED" if report["passed"] else "FAILED"
        logger.info(f"Validation {status} | errors={len(report['errors'])} | warnings={len(report['warnings'])}")
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
            logger.error(f"Failed to write validation log: {e}")