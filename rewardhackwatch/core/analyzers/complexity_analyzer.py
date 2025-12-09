"""Analyze code complexity in trajectories."""

import ast
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""

    cyclomatic_complexity: int
    lines_of_code: int
    function_count: int
    class_count: int
    import_count: int
    nesting_depth: int
    complexity_score: float  # 0-1 normalized


class ComplexityAnalyzer:
    """Analyze code complexity in agent outputs."""

    def __init__(self, max_complexity: int = 20, max_nesting: int = 5):
        self.max_complexity = max_complexity
        self.max_nesting = max_nesting

    def analyze(self, trajectory: dict[str, Any]) -> ComplexityResult:
        """Analyze complexity of code in trajectory."""
        code_outputs = trajectory.get("code_outputs", []) or []

        total_complexity = 0
        total_loc = 0
        total_functions = 0
        total_classes = 0
        total_imports = 0
        max_nesting = 0

        for code in code_outputs:
            if not isinstance(code, str):
                continue

            result = self._analyze_code(code)
            total_complexity += result.get("cyclomatic", 0)
            total_loc += result.get("loc", 0)
            total_functions += result.get("functions", 0)
            total_classes += result.get("classes", 0)
            total_imports += result.get("imports", 0)
            max_nesting = max(max_nesting, result.get("nesting", 0))

        # Normalize to 0-1 score (higher = more complex)
        complexity_score = min(1.0, total_complexity / self.max_complexity)

        return ComplexityResult(
            cyclomatic_complexity=total_complexity,
            lines_of_code=total_loc,
            function_count=total_functions,
            class_count=total_classes,
            import_count=total_imports,
            nesting_depth=max_nesting,
            complexity_score=complexity_score,
        )

    def _analyze_code(self, code: str) -> dict[str, int]:
        """Analyze a single code block."""
        result = {
            "cyclomatic": 1,  # Base complexity
            "loc": len([l for l in code.split("\n") if l.strip()]),
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "nesting": 0,
        }

        try:
            tree = ast.parse(code)
            result["cyclomatic"] = self._calculate_cyclomatic(tree)
            result["functions"] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            result["classes"] = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            result["imports"] = len(
                [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            )
            result["nesting"] = self._calculate_nesting(tree)
        except SyntaxError:
            # Fall back to regex-based analysis
            result["functions"] = len(re.findall(r"^def\s+\w+", code, re.MULTILINE))
            result["classes"] = len(re.findall(r"^class\s+\w+", code, re.MULTILINE))
            result["imports"] = len(re.findall(r"^(?:import|from)\s+", code, re.MULTILINE))

        return result

    def _calculate_cyclomatic(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _calculate_nesting(self, tree: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._calculate_nesting(node, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting(node, depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def is_suspiciously_simple(self, result: ComplexityResult) -> bool:
        """Check if code is suspiciously simple (potential hack)."""
        return (
            result.lines_of_code < 5
            and result.function_count == 0
            and result.complexity_score < 0.1
        )
