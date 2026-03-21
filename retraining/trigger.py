import json
import logging
import os
import urllib.request
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RetrainingTrigger:

    def __init__(self, config: Dict[str, Any]):
        retrain_cfg = config.get("retraining", {})
        self.token = os.getenv(retrain_cfg.get("github_token_env", "GITHUB_TOKEN"), "")
        self.repo_owner = retrain_cfg.get("repo_owner", "")
        self.repo_name = retrain_cfg.get("repo_name", "")
        self.workflow_id = retrain_cfg.get("workflow_id", "retrain.yml")
        self.branch = retrain_cfg.get("branch", "main")
        self.drift_trigger_count = retrain_cfg.get("drift_trigger_count", 2)
        self._consecutive_drift_count = 0

    def evaluate(self, alert_summary: Dict[str, Any]) -> bool:
        has_drift = any(
            a["type"] in ("feature_drift", "performance_degradation")
            for a in alert_summary.get("alerts", [])
        )

        if has_drift:
            self._consecutive_drift_count += 1
            logger.info(
                f"Consecutive drift count: {self._consecutive_drift_count}"
                f" / {self.drift_trigger_count}"
            )
        else:
            if self._consecutive_drift_count > 0:
                logger.info("Drift resolved — resetting counter.")
            self._consecutive_drift_count = 0
            return False

        if self._consecutive_drift_count >= self.drift_trigger_count:
            logger.warning(
                f"Drift persisted for {self.drift_trigger_count} consecutive batches."
                " Dispatching retraining workflow."
            )
            success = self._dispatch_github_workflow(reason="automated_drift_detection")
            if success:
                self._consecutive_drift_count = 0
            return success

        return False

    def _dispatch_github_workflow(self, reason: str = "automated_drift_detection") -> bool:
        if not self.token:
            logger.error(
                "GITHUB_TOKEN not set. Cannot trigger retraining. "
                "Set it as an env var or GitHub Actions secret."
            )
            return False

        url = (
            f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            f"/actions/workflows/{self.workflow_id}/dispatches"
        )
        payload = json.dumps({
            "ref": self.branch,
            "inputs": {
                "trigger_reason": reason,
                "triggered_at": datetime.utcnow().isoformat(),
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(f"Retraining workflow dispatched. HTTP {resp.status}")
                return resp.status == 204
        except urllib.error.HTTPError as e:
            logger.error(f"GitHub API error {e.code}: {e.read().decode()}")
            return False
        except Exception as e:
            logger.error(f"Failed to dispatch retraining workflow: {e}")
            return False