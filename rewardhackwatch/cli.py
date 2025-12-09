"""
RewardHackWatch CLI - Command-line interface for trajectory analysis.

Usage:
    rewardhackwatch analyze trajectory.json
    rewardhackwatch scan ./trajectories/
    rewardhackwatch dashboard
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="rewardhackwatch",
    help="Detect reward hacking → misalignment generalization in LLM agents",
)
console = Console()


def load_trajectory(path: Path) -> dict:
    """Load trajectory from JSON file."""
    with open(path) as f:
        return json.load(f)


def format_risk_level(risk: str) -> str:
    """Format risk level with color."""
    colors = {
        "none": "[green]NONE[/green]",
        "low": "[blue]LOW[/blue]",
        "medium": "[yellow]MEDIUM[/yellow]",
        "high": "[red]HIGH[/red]",
        "critical": "[bold red]CRITICAL[/bold red]",
    }
    return colors.get(risk.lower(), risk)


def format_score_bar(score: float, width: int = 10) -> str:
    """Format score as visual bar."""
    filled = int(score * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


@app.command()
def analyze(
    trajectory_path: Path = typer.Argument(..., help="Path to trajectory JSON file"),
    use_ml: bool = typer.Option(False, "--ml", help="Use ML detector (requires model)"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM judges"),
    llm_provider: str = typer.Option("claude", "--provider", help="LLM provider: claude or llama"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """
    Analyze a single trajectory for reward hacking and misalignment.
    """
    from rewardhackwatch.core import (
        ASTDetector,
        ClaudeJudge,
        GeneralizationTracker,
        LlamaJudge,
        MLDetector,
        PatternDetector,
    )

    console.print(
        Panel.fit(
            "[bold blue]RewardHackWatch[/bold blue] - Trajectory Analysis",
            subtitle=f"Analyzing: {trajectory_path.name}",
        )
    )

    # Load trajectory
    if not trajectory_path.exists():
        console.print(f"[red]Error: File not found: {trajectory_path}[/red]")
        raise typer.Exit(1)

    trajectory = load_trajectory(trajectory_path)

    results = {
        "file": str(trajectory_path),
        "detectors": {},
        "judges": {},
        "generalization": {},
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Pattern Detection
        task = progress.add_task("Running pattern detector...", total=1)
        pattern_detector = PatternDetector()
        pattern_result = pattern_detector.detect(trajectory)
        results["detectors"]["pattern"] = pattern_result.to_dict()
        progress.update(task, completed=1)

        # AST Detection
        task = progress.add_task("Running AST detector...", total=1)
        ast_detector = ASTDetector()
        ast_result = ast_detector.detect(trajectory)
        results["detectors"]["ast"] = ast_result.to_dict()
        progress.update(task, completed=1)

        # ML Detection (optional)
        if use_ml:
            task = progress.add_task("Running ML detector...", total=1)
            try:
                ml_detector = MLDetector()
                ml_result = ml_detector.detect(trajectory)
                results["detectors"]["ml"] = ml_result.to_dict()
            except Exception as e:
                results["detectors"]["ml"] = {"error": str(e)}
            progress.update(task, completed=1)

        # LLM Judges (optional)
        if use_llm:
            task = progress.add_task(f"Running {llm_provider} judge...", total=1)
            try:
                if llm_provider == "claude":
                    judge = ClaudeJudge()
                else:
                    judge = LlamaJudge()

                judge_result = judge.judge_sync(trajectory)
                results["judges"][llm_provider] = judge_result.to_dict()
            except Exception as e:
                results["judges"][llm_provider] = {"error": str(e)}
            progress.update(task, completed=1)

        # Generalization Tracking
        task = progress.add_task("Analyzing generalization...", total=1)
        tracker = GeneralizationTracker()

        # Build scores from detector results
        hack_scores = []
        misalign_scores = []

        if "steps" in trajectory:
            for i, step in enumerate(trajectory["steps"]):
                # Use detector scores if available
                hack_scores.append(pattern_result.score)
                misalign_scores.append(pattern_result.score * 0.8)  # Approximate
                tracker.update(hack_scores[-1], misalign_scores[-1])

        gen_result = tracker.analyze()
        results["generalization"] = gen_result.to_dict()
        progress.update(task, completed=1)

    # Display results
    console.print()

    # Scores table
    table = Table(title="Detection Scores", show_header=True)
    table.add_column("Detector", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Risk Level", justify="center")
    table.add_column("Detections", justify="right")

    table.add_row(
        "Pattern",
        f"{pattern_result.score:.2f} {format_score_bar(pattern_result.score)}",
        format_risk_level(pattern_result.risk_level.value),
        str(len(pattern_result.detections)),
    )

    table.add_row(
        "AST",
        f"{ast_result.score:.2f} {format_score_bar(ast_result.score)}",
        format_risk_level(ast_result.risk_level.value),
        str(len(ast_result.detections)),
    )

    if use_ml and "ml" in results["detectors"] and "error" not in results["detectors"]["ml"]:
        ml_data = results["detectors"]["ml"]
        table.add_row(
            "ML",
            f"{ml_data['score']:.2f} {format_score_bar(ml_data['score'])}",
            format_risk_level(ml_data["risk_level"]),
            str(len(ml_data.get("detections", []))),
        )

    console.print(table)
    console.print()

    # Generalization result
    gen_panel = Panel(
        f"""[bold]Generalization Analysis[/bold]

Correlation: {gen_result.correlation:.2f}
Detected: {"[red]YES[/red]" if gen_result.generalization_detected else "[green]NO[/green]"}
Risk Level: {format_risk_level(gen_result.risk_level)}
Transition Points: {len(gen_result.transition_points)}
""",
        title="🔍 Generalization Tracking",
        border_style="yellow" if gen_result.generalization_detected else "green",
    )
    console.print(gen_panel)

    # Show detections if verbose
    if verbose and (pattern_result.detections or ast_result.detections):
        console.print()
        console.print("[bold]Flagged Patterns:[/bold]")
        for d in pattern_result.detections[:10]:
            console.print(f"  • [{d.location}] {d.pattern_name}: {d.description}")
        for d in ast_result.detections[:10]:
            console.print(f"  • [{d.location}] {d.pattern_name}: {d.description}")

    # Save output
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to: {output}[/green]")

    # Exit code based on risk
    if gen_result.risk_level in ("high", "critical"):
        raise typer.Exit(1)


@app.command()
def scan(
    directory: Path = typer.Argument(..., help="Directory containing trajectory files"),
    pattern: str = typer.Option("*.json", "--pattern", "-p", help="File pattern to match"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
):
    """
    Scan multiple trajectories in a directory.
    """
    from rewardhackwatch.core import ASTDetector, PatternDetector

    console.print(
        Panel.fit(
            "[bold blue]RewardHackWatch[/bold blue] - Batch Scan",
            subtitle=f"Directory: {directory}",
        )
    )

    if not directory.exists():
        console.print(f"[red]Error: Directory not found: {directory}[/red]")
        raise typer.Exit(1)

    files = list(directory.glob(pattern))
    if not files:
        console.print(f"[yellow]No files found matching pattern: {pattern}[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(files)} files to analyze\n")

    pattern_detector = PatternDetector()
    ast_detector = ASTDetector()

    results = []
    high_risk_count = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Scanning...", total=len(files))

        for file_path in files:
            try:
                trajectory = load_trajectory(file_path)
                pattern_result = pattern_detector.detect(trajectory)
                ast_result = ast_detector.detect(trajectory)

                combined_score = (pattern_result.score + ast_result.score) / 2
                risk = pattern_result.risk_level.value

                if risk in ("high", "critical"):
                    high_risk_count += 1

                results.append(
                    {
                        "file": str(file_path),
                        "score": combined_score,
                        "risk": risk,
                        "detections": len(pattern_result.detections) + len(ast_result.detections),
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "file": str(file_path),
                        "error": str(e),
                    }
                )

            progress.update(task, advance=1)

    # Summary table
    console.print()
    table = Table(title="Scan Results", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Risk", justify="center")
    table.add_column("Detections", justify="right")

    for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
        if "error" in r:
            table.add_row(Path(r["file"]).name, "[red]ERROR[/red]", "-", "-")
        else:
            table.add_row(
                Path(r["file"]).name,
                f"{r['score']:.2f}",
                format_risk_level(r["risk"]),
                str(r["detections"]),
            )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {high_risk_count}/{len(files)} high-risk trajectories")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to: {output}[/green]")


@app.command()
def dashboard():
    """
    Start the Streamlit dashboard.
    """
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        console.print("[red]Dashboard not found. Please ensure dashboard/app.py exists.[/red]")
        raise typer.Exit(1)

    console.print("[bold blue]Starting RewardHackWatch Dashboard...[/bold blue]")
    console.print("Open http://localhost:8501 in your browser\n")

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


@app.command()
def version():
    """Show version information."""
    from rewardhackwatch import __version__

    console.print(f"RewardHackWatch v{__version__}")


if __name__ == "__main__":
    app()
