"""Discord integration for alerts."""

from typing import Any

from .webhook import WebhookNotifier


class DiscordNotifier(WebhookNotifier):
    """Send alerts to Discord via webhook."""

    def __init__(self, webhook_url: str, username: str = "RewardHackWatch"):
        super().__init__(webhook_url)
        self.username = username

    def format_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Format alert for Discord embed."""
        score = alert.get("score", 0)
        level = alert.get("level", "unknown")
        message = alert.get("message", "No message")

        # Choose color based on score (Discord uses decimal colors)
        if score >= 0.8:
            color = 14423100  # Red
        elif score >= 0.6:
            color = 16744448  # Orange
        elif score >= 0.4:
            color = 16776960  # Yellow
        else:
            color = 5763719  # Green

        return {
            "username": self.username,
            "embeds": [
                {
                    "title": "RewardHackWatch Alert",
                    "color": color,
                    "fields": [
                        {"name": "Risk Score", "value": f"{score:.1%}", "inline": True},
                        {"name": "Risk Level", "value": level.upper(), "inline": True},
                        {
                            "name": "Details",
                            "value": message[:1024],  # Discord limit
                            "inline": False,
                        },
                    ],
                    "footer": {"text": "RewardHackWatch v0.1.0"},
                }
            ],
        }

    def notify(self, alert: dict[str, Any]) -> bool:
        """Send formatted alert to Discord."""
        formatted = self.format_alert(alert)
        return super().notify(formatted)

    def send_simple(self, content: str) -> bool:
        """Send simple text message."""
        payload = {"username": self.username, "content": content}
        return super().notify(payload)

    def send_summary(self, results: list[dict[str, Any]]) -> bool:
        """Send summary of analysis results."""
        if not results:
            return self.send_simple("No results to report.")

        scores = [r.get("score", 0) for r in results]
        high_risk = sum(1 for s in scores if s >= 0.6)

        summary = (
            f"**Analysis Summary**\n"
            f"Trajectories analyzed: {len(results)}\n"
            f"High risk detected: {high_risk}\n"
            f"Max score: {max(scores):.1%}\n"
            f"Average score: {sum(scores) / len(scores):.1%}"
        )
        return self.send_simple(summary)
