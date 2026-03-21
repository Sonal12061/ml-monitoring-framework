import json
import logging
import smtplib
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AlertEngine:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        alert_cfg = config.get("alerting", {})
        self.email_cfg = alert_cfg.get("email", {})
        self.webhook_cfg = alert_cfg.get("webhook", {})
        self.log_path = alert_cfg.get("log_path", "logs/alerts.json")

    def evaluate_and_alert(
        self,
        validation_report: Optional[Dict] = None,
        drift_report: Optional[Dict] = None,
        performance_report: Optional[Dict] = None,
    ) -> Dict[str, Any]:

        alerts = []

        # Validation failures
        if validation_report and not validation_report.get("passed", True):
            alerts.append({
                "type": "data_validation_failure",
                "severity": "critical",
                "details": validation_report.get("errors", []),
            })

        # Drift detected
        if drift_report and drift_report.get("overall_drift_detected"):
            drifted = [
                f"{feat} ({info.get('method', '?')})"
                for feat, info in drift_report.get("features", {}).items()
                if info.get("drifted")
            ]
            if drift_report.get("prediction_drift") and \
               drift_report["prediction_drift"].get("drifted"):
                drifted.append("prediction_score (ks_test)")

            alerts.append({
                "type": "feature_drift",
                "severity": "high",
                "details": f"Drift detected in: {drifted}",
            })

        # Performance degradation
        if performance_report and performance_report.get("performance_alert"):
            degraded = [
                f"{k}: {v['current']:.4f} vs baseline {v['baseline']:.4f} "
                f"(drop={v['drop']:.4f}, threshold={v['threshold']:.4f})"
                for k, v in performance_report.get("degradation_check", {}).items()
                if v.get("alert")
            ]
            alerts.append({
                "type": "performance_degradation",
                "severity": "high",
                "details": degraded,
            })

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_alerts": len(alerts),
            "alerts": alerts,
            "action_required": len(alerts) > 0,
        }

        if alerts:
            logger.warning(f"[ALERT] {len(alerts)} issue(s) detected — firing notifications.")
            self._send_email(summary)
            self._send_webhook(summary)
        else:
            logger.info("All monitoring checks passed — no alerts.")

        self._log_alert(summary)
        return summary

    def _send_email(self, summary: Dict) -> None:
        cfg = self.email_cfg
        if not cfg.get("enabled"):
            return
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[ML Monitor] {summary['total_alerts']} Alert(s) — Action Required"
            msg["From"] = cfg["sender"]
            msg["To"] = ", ".join(cfg["recipients"])
            msg.attach(MIMEText(self._format_html_email(summary), "html"))

            with smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"]) as server:
                server.starttls()
                server.login(cfg["sender"], cfg.get("password", ""))
                server.sendmail(cfg["sender"], cfg["recipients"], msg.as_string())
            logger.info("Alert email sent.")
        except Exception as e:
            logger.error(f"Email dispatch failed: {e}")

    def _send_webhook(self, summary: Dict) -> None:
        cfg = self.webhook_cfg
        if not cfg.get("enabled") or not cfg.get("url"):
            return
        try:
            payload = json.dumps({
                "text": (
                    f":rotating_light: *ML Monitor Alert* — "
                    f"{summary['total_alerts']} issue(s) at {summary['timestamp']}"
                ),
                "attachments": [
                    {"color": "danger", "text": f"{a['type']}: {a['details']}"}
                    for a in summary["alerts"]
                ],
            }).encode("utf-8")

            req = urllib.request.Request(
                cfg["url"],
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info("Webhook alert sent.")
        except Exception as e:
            logger.error(f"Webhook dispatch failed: {e}")

    def _format_html_email(self, summary: Dict) -> str:
        rows = "".join(
            f"<tr><td>{a['type']}</td><td>{a['severity']}</td>"
            f"<td>{a['details']}</td></tr>"
            for a in summary["alerts"]
        )
        return f"""
        <html><body style="font-family: Arial, sans-serif;">
          <h2 style="color: #c0392b;">ML Monitoring Alert</h2>
          <p><strong>Timestamp:</strong> {summary['timestamp']}</p>
          <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
            <tr style="background:#f2f2f2;">
              <th>Type</th><th>Severity</th><th>Details</th>
            </tr>
            {rows}
          </table>
        </body></html>
        """

    def _log_alert(self, summary: Dict) -> None:
        try:
            try:
                with open(self.log_path) as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
            logs.append(summary)
            with open(self.log_path, "w") as f:
                json.dump(logs, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write alert log: {e}")