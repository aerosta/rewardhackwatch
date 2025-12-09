"""Webhook integration for alerts."""

import json
import urllib.error
import urllib.request
from typing import Any, Optional


class WebhookNotifier:
    """Send alerts via webhook."""

    def __init__(self, url: str, headers: Optional[dict[str, str]] = None, timeout: int = 10):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def notify(self, alert: dict[str, Any]) -> bool:
        """Send alert to webhook.

        Args:
            alert: Alert data to send

        Returns:
            True if successful, False otherwise
        """
        try:
            data = json.dumps(alert).encode("utf-8")
            request = urllib.request.Request(
                self.url, data=data, headers=self.headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            return False

    def test_connection(self) -> bool:
        """Test webhook connection."""
        test_payload = {"type": "test", "message": "RewardHackWatch webhook test"}
        return self.notify(test_payload)

    def notify_batch(self, alerts: list) -> dict[str, int]:
        """Send multiple alerts.

        Returns:
            Dict with 'success' and 'failed' counts
        """
        results = {"success": 0, "failed": 0}
        for alert in alerts:
            if self.notify(alert):
                results["success"] += 1
            else:
                results["failed"] += 1
        return results
