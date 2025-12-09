"""Generate HTML reports from analysis results."""

from datetime import datetime
from pathlib import Path
from typing import Any


class HTMLReporter:
    """Generate HTML analysis reports."""

    def __init__(self, title: str = "RewardHackWatch Report"):
        self.title = title

    def generate(self, results: dict[str, Any]) -> str:
        """Generate HTML report."""
        score = results.get("score", 0)
        score_class = self._get_score_class(score)
        detections = results.get("detections", [])

        detections_html = self._render_detections(detections)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .timestamp {{ color: #666; font-size: 14px; }}
        .score {{
            font-size: 48px;
            font-weight: bold;
            margin: 20px 0;
        }}
        .critical {{ color: #dc3545; }}
        .high {{ color: #fd7e14; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #28a745; }}
        .none {{ color: #6c757d; }}
        .detections {{
            margin-top: 30px;
        }}
        .detection {{
            background: #f8f9fa;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 4px 4px 0;
        }}
        .detection-title {{
            font-weight: bold;
            color: #333;
        }}
        .detection-desc {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Overall Risk Score</h2>
        <p class="score {score_class}">{score:.1%}</p>

        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Risk Level</td>
                <td class="{score_class}">{results.get("risk_level", "Unknown").upper()}</td>
            </tr>
            <tr>
                <td>Detections</td>
                <td>{len(detections)}</td>
            </tr>
            <tr>
                <td>Generalization Risk</td>
                <td>{results.get("generalization_risk", 0):.1%}</td>
            </tr>
        </table>

        <div class="detections">
            <h2>Detections ({len(detections)})</h2>
            {detections_html}
        </div>
    </div>
</body>
</html>
"""

    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score."""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        return "none"

    def _render_detections(self, detections: list[dict[str, Any]]) -> str:
        """Render detections as HTML."""
        if not detections:
            return "<p>No detections found.</p>"

        html_parts = []
        for d in detections:
            pattern = d.get("pattern", d.get("pattern_name", "Unknown"))
            desc = d.get("description", "No description")
            location = d.get("location", "Unknown location")

            html_parts.append(f"""
            <div class="detection">
                <div class="detection-title">{pattern}</div>
                <div class="detection-desc">{desc}</div>
                <div class="detection-desc">Location: {location}</div>
            </div>
            """)

        return "\n".join(html_parts)

    def save(self, results: dict[str, Any], path: str):
        """Save HTML report to file."""
        html = self.generate(results)
        Path(path).write_text(html)
