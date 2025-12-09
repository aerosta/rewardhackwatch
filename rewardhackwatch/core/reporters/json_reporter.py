"""Generate JSON reports from analysis results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONReporter:
    """Generate JSON analysis reports."""

    def __init__(self, pretty: bool = True, include_metadata: bool = True):
        self.pretty = pretty
        self.include_metadata = include_metadata

    def generate(self, results: dict[str, Any]) -> str:
        """Generate JSON report."""
        report = {"version": "1.0", "results": results}

        if self.include_metadata:
            report["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "generator": "RewardHackWatch",
                "generator_version": "0.1.0",
            }

        indent = 2 if self.pretty else None
        return json.dumps(report, indent=indent, default=str)

    def save(self, results: dict[str, Any], path: str):
        """Save JSON report to file."""
        Path(path).write_text(self.generate(results))

    def generate_summary(self, results: dict[str, Any]) -> str:
        """Generate summary-only JSON."""
        summary = {
            "score": results.get("score", 0),
            "risk_level": results.get("risk_level", "unknown"),
            "detection_count": len(results.get("detections", [])),
            "generated_at": datetime.now().isoformat(),
        }
        return json.dumps(summary, indent=2 if self.pretty else None)

    def generate_batch(self, results_list: list) -> str:
        """Generate JSON for batch results."""
        batch_report = {
            "version": "1.0",
            "batch_size": len(results_list),
            "results": results_list,
            "summary": self._compute_batch_summary(results_list),
        }

        if self.include_metadata:
            batch_report["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "generator": "RewardHackWatch",
            }

        indent = 2 if self.pretty else None
        return json.dumps(batch_report, indent=indent, default=str)

    def _compute_batch_summary(self, results_list: list) -> dict[str, Any]:
        """Compute summary statistics for batch."""
        if not results_list:
            return {}

        scores = [r.get("score", 0) for r in results_list]
        return {
            "count": len(results_list),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "high_risk_count": sum(1 for s in scores if s >= 0.6),
        }
