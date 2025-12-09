# Contributing to RewardHackWatch

Thank you for your interest in contributing to RewardHackWatch!

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes and add tests**
4. **Run the test suite**: `pytest tests/`
5. **Submit a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-fork/rewardhackwatch.git
cd rewardhackwatch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Code Style

- Use **Black** for formatting: `black .`
- Use **Ruff** for linting: `ruff check .`
- Add **type hints** to all functions
- Write **docstrings** for public functions

## Testing Guidelines

- Add tests for new features
- Maintain or improve test coverage
- Use `pytest` fixtures for common setup

## Pull Request Process

1. Update the README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Request review from maintainers

## Questions?

Feel free to open an issue for questions or discussion!
