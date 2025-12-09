#!/usr/bin/env python3
"""Custom pattern detection example for RewardHackWatch.

This example shows how to extend the detection system with custom patterns.
"""

from rewardhackwatch.core.detectors.pattern_detector import PatternDetector


def main():
    print("=" * 60)
    print("Custom Pattern Detection Example")
    print("=" * 60)

    # Create a pattern detector
    detector = PatternDetector()

    # Show built-in patterns
    print(f"\nBuilt-in patterns: {len(detector.patterns)}")
    print("\nPattern categories:")
    categories = {}
    for pattern in detector.patterns:
        cat = pattern.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} patterns")

    # Add custom patterns
    print("\n" + "=" * 60)
    print("Adding Custom Patterns")
    print("=" * 60)

    custom_patterns = [
        {
            "pattern": r"eval\s*\(",
            "category": "code_injection",
            "severity": 0.9,
            "description": "Potential code injection via eval()",
        },
        {
            "pattern": r"__import__\s*\(",
            "category": "dynamic_import",
            "severity": 0.7,
            "description": "Dynamic import detected",
        },
        {
            "pattern": r"subprocess\.call|os\.system",
            "category": "shell_execution",
            "severity": 0.8,
            "description": "Shell command execution",
        },
    ]

    for pattern in custom_patterns:
        detector.add_pattern(pattern)
        print(f"Added pattern: {pattern['category']}")

    print(f"\nTotal patterns after adding custom: {len(detector.patterns)}")

    # Test custom patterns
    print("\n" + "=" * 60)
    print("Testing Custom Patterns")
    print("=" * 60)

    test_cases = [
        "result = eval(user_input)",
        "module = __import__('os')",
        "subprocess.call(['rm', '-rf', '/'])",
        "print('Hello, World!')",
    ]

    for code in test_cases:
        result = detector.detect(code)
        status = "DETECTED" if result.is_suspicious else "CLEAN"
        print(f"\n{status}: {code[:50]}...")
        if result.matches:
            for match in result.matches:
                print(f"  -> {match.get('category')}: {match.get('pattern')}")


if __name__ == "__main__":
    main()
