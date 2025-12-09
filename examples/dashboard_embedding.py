#!/usr/bin/env python3
"""
Dashboard Embedding Example

Demonstrates how to embed RewardHackWatch visualizations and
monitoring components into custom applications.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AlertConfig:
    """Configuration for alert thresholds."""

    hack_score_warning: float = 0.5
    hack_score_critical: float = 0.8
    rmgi_warning: float = 0.5
    rmgi_critical: float = 0.7
    max_alerts_per_minute: int = 10


class MonitoringWidget:
    """
    A widget for monitoring reward hacking in real-time.

    Can be embedded in web applications or used programmatically.
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize the monitoring widget.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        self.history: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []

    def add_result(self, result: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Add an analysis result and check for alerts.

        Args:
            result: Analysis result dictionary

        Returns:
            Alert dictionary if threshold exceeded, None otherwise
        """
        self.history.append(result)

        alert = None
        hack_score = result.get("hack_score", 0)

        if hack_score >= self.config.hack_score_critical:
            alert = {
                "level": "critical",
                "message": f"Critical hack score: {hack_score:.2f}",
                "result": result,
            }
        elif hack_score >= self.config.hack_score_warning:
            alert = {
                "level": "warning",
                "message": f"Elevated hack score: {hack_score:.2f}",
                "result": result,
            }

        if alert:
            self.alerts.append(alert)

        return alert

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of monitoring statistics.

        Returns:
            Summary dictionary
        """
        if not self.history:
            return {"total": 0, "hacks": 0, "hack_rate": 0.0}

        total = len(self.history)
        hacks = sum(1 for r in self.history if r.get("is_hack", False))
        scores = [r.get("hack_score", 0) for r in self.history]

        return {
            "total": total,
            "hacks": hacks,
            "hack_rate": hacks / total,
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "critical_alerts": len([a for a in self.alerts if a["level"] == "critical"]),
            "warning_alerts": len([a for a in self.alerts if a["level"] == "warning"]),
        }

    def render_text(self) -> str:
        """
        Render the widget as text for CLI display.

        Returns:
            Text representation of the monitoring state
        """
        summary = self.get_summary()

        lines = [
            "╔════════════════════════════════════════╗",
            "║     RewardHackWatch Monitor            ║",
            "╠════════════════════════════════════════╣",
            f"║ Total Analyzed: {summary['total']:>20} ║",
            f"║ Hacks Detected: {summary['hacks']:>20} ║",
            f"║ Hack Rate: {summary['hack_rate']:>24.1%} ║",
            f"║ Avg Score: {summary['avg_score']:>24.2f} ║",
            "╠════════════════════════════════════════╣",
            f"║ 🔴 Critical Alerts: {summary['critical_alerts']:>16} ║",
            f"║ 🟡 Warning Alerts: {summary['warning_alerts']:>17} ║",
            "╚════════════════════════════════════════╝",
        ]

        return "\n".join(lines)

    def render_html(self) -> str:
        """
        Render the widget as HTML for web embedding.

        Returns:
            HTML string
        """
        summary = self.get_summary()

        return f"""
        <div class="rhw-monitor">
            <h3>RewardHackWatch Monitor</h3>
            <div class="stats">
                <div class="stat">
                    <span class="label">Total</span>
                    <span class="value">{summary["total"]}</span>
                </div>
                <div class="stat">
                    <span class="label">Hacks</span>
                    <span class="value">{summary["hacks"]}</span>
                </div>
                <div class="stat">
                    <span class="label">Rate</span>
                    <span class="value">{summary["hack_rate"]:.1%}</span>
                </div>
            </div>
            <div class="alerts">
                <span class="critical">🔴 {summary["critical_alerts"]}</span>
                <span class="warning">🟡 {summary["warning_alerts"]}</span>
            </div>
        </div>
        """

    def render_json(self) -> str:
        """
        Render the widget state as JSON for API responses.

        Returns:
            JSON string
        """
        return json.dumps(
            {
                "summary": self.get_summary(),
                "recent_alerts": self.alerts[-10:],
                "config": {
                    "hack_score_warning": self.config.hack_score_warning,
                    "hack_score_critical": self.config.hack_score_critical,
                },
            },
            indent=2,
        )


class RMGIChart:
    """
    Generates RMGI tracking charts.

    Can be rendered as ASCII, HTML/SVG, or data for charting libraries.
    """

    def __init__(self, max_points: int = 50):
        """
        Initialize the RMGI chart.

        Args:
            max_points: Maximum number of points to display
        """
        self.max_points = max_points
        self.data: list[float] = []

    def add_point(self, rmgi: float) -> None:
        """Add an RMGI data point."""
        self.data.append(rmgi)
        if len(self.data) > self.max_points:
            self.data.pop(0)

    def render_ascii(self, width: int = 50, height: int = 10) -> str:
        """
        Render as ASCII chart.

        Args:
            width: Chart width in characters
            height: Chart height in lines

        Returns:
            ASCII chart string
        """
        if not self.data:
            return "No data"

        # Normalize data to chart height
        min_val, max_val = 0.0, 1.0

        lines = []
        for row in range(height, 0, -1):
            threshold = (row / height) * (max_val - min_val) + min_val
            line = "│"
            for i, val in enumerate(self.data[-width:]):
                if val >= threshold:
                    # Color based on value
                    if val >= 0.7:
                        line += "█"  # Critical
                    elif val >= 0.5:
                        line += "▓"  # Warning
                    else:
                        line += "▒"  # Normal
                else:
                    line += " "
            lines.append(line)

        # Add axis
        lines.append("└" + "─" * min(len(self.data), width))
        lines.insert(0, "RMGI (0.0 - 1.0)")

        return "\n".join(lines)

    def render_data(self) -> dict[str, Any]:
        """
        Render as data for charting libraries.

        Returns:
            Data dictionary for Chart.js, Plotly, etc.
        """
        return {
            "labels": list(range(len(self.data))),
            "datasets": [
                {
                    "label": "RMGI",
                    "data": self.data,
                    "borderColor": "rgb(75, 192, 192)",
                    "fill": False,
                }
            ],
            "thresholds": [
                {"value": 0.7, "color": "red", "label": "Critical"},
                {"value": 0.5, "color": "orange", "label": "Warning"},
            ],
        }


def main():
    """Demonstrate dashboard embedding components."""
    print("=" * 60)
    print("DASHBOARD EMBEDDING DEMONSTRATION")
    print("=" * 60)

    # Initialize monitoring widget
    widget = MonitoringWidget()

    # Simulate some analysis results
    sample_results = [
        {"task_id": "001", "is_hack": False, "hack_score": 0.1},
        {"task_id": "002", "is_hack": False, "hack_score": 0.2},
        {"task_id": "003", "is_hack": False, "hack_score": 0.3},
        {"task_id": "004", "is_hack": True, "hack_score": 0.6},  # Warning
        {"task_id": "005", "is_hack": True, "hack_score": 0.85},  # Critical
        {"task_id": "006", "is_hack": False, "hack_score": 0.15},
        {"task_id": "007", "is_hack": True, "hack_score": 0.9},  # Critical
    ]

    print("\n--- Processing Results ---")
    for result in sample_results:
        alert = widget.add_result(result)
        if alert:
            print(f"⚠️ {alert['level'].upper()}: {alert['message']}")

    # Display text widget
    print("\n--- Text Widget ---")
    print(widget.render_text())

    # Display JSON
    print("\n--- JSON Output ---")
    print(widget.render_json())

    # RMGI Chart
    print("\n--- RMGI Chart ---")
    chart = RMGIChart()

    # Simulate RMGI values
    import random

    random.seed(42)
    for i in range(30):
        if i < 15:
            rmgi = random.uniform(0.1, 0.4)
        else:
            rmgi = random.uniform(0.5, 0.9)
        chart.add_point(rmgi)

    print(chart.render_ascii(width=30, height=8))

    print("\n" + "=" * 60)
    print("EMBEDDING IN APPLICATIONS")
    print("=" * 60)
    print("""
To embed in your application:

1. Web (Flask/Django):

   from examples.dashboard_embedding import MonitoringWidget

   @app.route('/monitor')
   def monitor():
       return widget.render_html()

2. API Response:

   @app.route('/api/monitor')
   def monitor_api():
       return json.loads(widget.render_json())

3. CLI Tool:

   print(widget.render_text())

4. Charting Library:

   chart_data = chart.render_data()
   # Use with Chart.js, Plotly, etc.
""")


if __name__ == "__main__":
    main()
