"""Slack integration for alerts."""

from typing import Any, Optional

from .webhook import WebhookNotifier


class SlackNotifier(WebhookNotifier):
    """Send alerts to Slack via incoming webhook."""

    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        super().__init__(webhook_url)
        self.channel = channel

    def format_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Format alert for Slack Block Kit."""
        score = alert.get("score", 0)
        level = alert.get("level", "unknown")
        message = alert.get("message", "No message")

        # Choose emoji based on score
        if score >= 0.8:
            emoji = ":red_circle:"
            color = "#dc3545"
        elif score >= 0.6:
            emoji = ":large_orange_circle:"
            color = "#fd7e14"
        elif score >= 0.4:
            emoji = ":large_yellow_circle:"
            color = "#ffc107"
        else:
            emoji = ":large_green_circle:"
            color = "#28a745"

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} RewardHackWatch Alert",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Risk Score:*\n{score:.1%}"},
                        {"type": "mrkdwn", "text": f"*Risk Level:*\n{level.upper()}"},
                    ],
                },
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Details:*\n{message}"}},
            ],
            "attachments": [
                {"color": color, "fallback": f"RewardHackWatch Alert: {level} - Score: {score:.1%}"}
            ],
        }

        if self.channel:
            payload["channel"] = self.channel

        return payload

    def notify(self, alert: dict[str, Any]) -> bool:
        """Send formatted alert to Slack."""
        formatted = self.format_alert(alert)
        return super().notify(formatted)

    def send_summary(self, results: list[dict[str, Any]]) -> bool:
        """Send summary of multiple results."""
        if not results:
            return True

        scores = [r.get("score", 0) for r in results]
        high_risk = sum(1 for s in scores if s >= 0.6)

        summary = {
            "score": max(scores),
            "level": "high" if high_risk > 0 else "low",
            "message": f"Analyzed {len(results)} trajectories. {high_risk} high-risk detected. Max score: {max(scores):.1%}",
        }
        return self.notify(summary)
