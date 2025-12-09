"""Email integration for alerts."""

import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional


@dataclass
class EmailConfig:
    """Email configuration."""

    smtp_host: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    from_name: str = "RewardHackWatch"


class EmailNotifier:
    """Send alerts via email."""

    def __init__(self, config: EmailConfig):
        self.config = config

    def notify(
        self, alert: dict[str, Any], recipients: list[str], subject: Optional[str] = None
    ) -> bool:
        """Send alert email.

        Args:
            alert: Alert data
            recipients: List of email addresses
            subject: Optional custom subject

        Returns:
            True if successful
        """
        if not recipients:
            return False

        try:
            msg = self._create_message(alert, recipients, subject)

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.username, self.config.password)
                server.sendmail(self.config.username, recipients, msg.as_string())
            return True
        except Exception:
            return False

    def _create_message(
        self, alert: dict[str, Any], recipients: list[str], subject: Optional[str]
    ) -> MIMEMultipart:
        """Create email message."""
        score = alert.get("score", 0)
        level = alert.get("level", "unknown")

        if subject is None:
            subject = f"[{level.upper()}] RewardHackWatch Alert - Score: {score:.1%}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.from_name} <{self.config.username}>"
        msg["To"] = ", ".join(recipients)

        # Plain text version
        text_content = self._format_text(alert)
        msg.attach(MIMEText(text_content, "plain"))

        # HTML version
        html_content = self._format_html(alert)
        msg.attach(MIMEText(html_content, "html"))

        return msg

    def _format_text(self, alert: dict[str, Any]) -> str:
        """Format alert as plain text."""
        score = alert.get("score", 0)
        level = alert.get("level", "unknown")
        message = alert.get("message", "No details")

        return f"""RewardHackWatch Alert

Risk Score: {score:.1%}
Risk Level: {level.upper()}

Details:
{message}

---
This is an automated message from RewardHackWatch.
"""

    def _format_html(self, alert: dict[str, Any]) -> str:
        """Format alert as HTML."""
        score = alert.get("score", 0)
        level = alert.get("level", "unknown")
        message = alert.get("message", "No details")

        color = "#dc3545" if score >= 0.6 else "#ffc107" if score >= 0.4 else "#28a745"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background: {color}; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .score {{ font-size: 24px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RewardHackWatch Alert</h1>
    </div>
    <div class="content">
        <p class="score">Risk Score: {score:.1%}</p>
        <p><strong>Risk Level:</strong> {level.upper()}</p>
        <p><strong>Details:</strong></p>
        <p>{message}</p>
    </div>
</body>
</html>
"""

    def test_connection(self, test_recipient: str) -> bool:
        """Test email connection."""
        test_alert = {
            "score": 0.0,
            "level": "test",
            "message": "This is a test message from RewardHackWatch.",
        }
        return self.notify(test_alert, [test_recipient], "RewardHackWatch Test")
