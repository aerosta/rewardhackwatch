"""Tests for integration modules."""

from unittest.mock import MagicMock, patch


class TestWebhookNotifier:
    """Tests for webhook notifier."""

    def test_format_alert(self):
        """Test alert formatting."""
        from rewardhackwatch.integrations.webhook import WebhookNotifier

        notifier = WebhookNotifier("http://example.com/webhook")
        # WebhookNotifier doesn't format, just sends as-is
        # Just verify it doesn't crash
        assert notifier.url == "http://example.com/webhook"

    @patch("urllib.request.urlopen")
    def test_notify_success(self, mock_urlopen):
        """Test successful notification."""
        from rewardhackwatch.integrations.webhook import WebhookNotifier

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        notifier = WebhookNotifier("http://example.com/webhook")
        result = notifier.notify({"score": 0.75})
        assert result


class TestSlackNotifier:
    """Tests for Slack notifier."""

    def test_format_alert(self):
        """Test Slack alert formatting."""
        from rewardhackwatch.integrations.slack import SlackNotifier

        notifier = SlackNotifier("http://hooks.slack.com/test")
        alert = {"score": 0.8, "level": "high", "message": "Test alert"}

        formatted = notifier.format_alert(alert)
        assert "blocks" in formatted
        assert len(formatted["blocks"]) > 0

    def test_format_alert_colors(self):
        """Test color selection based on score."""
        from rewardhackwatch.integrations.slack import SlackNotifier

        notifier = SlackNotifier("http://hooks.slack.com/test")

        high_alert = notifier.format_alert({"score": 0.9})
        low_alert = notifier.format_alert({"score": 0.2})

        # Both should have attachments with colors
        assert "attachments" in high_alert
        assert "attachments" in low_alert


class TestDiscordNotifier:
    """Tests for Discord notifier."""

    def test_format_alert(self):
        """Test Discord alert formatting."""
        from rewardhackwatch.integrations.discord import DiscordNotifier

        notifier = DiscordNotifier("http://discord.com/webhook")
        alert = {"score": 0.75, "level": "high", "message": "Test"}

        formatted = notifier.format_alert(alert)
        assert "embeds" in formatted
        assert len(formatted["embeds"]) == 1

    def test_username_customization(self):
        """Test custom username."""
        from rewardhackwatch.integrations.discord import DiscordNotifier

        notifier = DiscordNotifier("http://discord.com/webhook", username="CustomBot")
        formatted = notifier.format_alert({"score": 0.5})
        assert formatted["username"] == "CustomBot"


class TestEmailNotifier:
    """Tests for email notifier."""

    def test_config_dataclass(self):
        """Test email config dataclass."""
        from rewardhackwatch.integrations.email_notifier import EmailConfig

        config = EmailConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="secret",
        )
        assert config.use_tls
        assert config.from_name == "RewardHackWatch"

    def test_format_text(self):
        """Test text formatting."""
        from rewardhackwatch.integrations.email_notifier import EmailConfig, EmailNotifier

        config = EmailConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="secret",
        )
        notifier = EmailNotifier(config)

        alert = {"score": 0.75, "level": "high", "message": "Test"}
        text = notifier._format_text(alert)

        assert "75.0%" in text
        assert "HIGH" in text
