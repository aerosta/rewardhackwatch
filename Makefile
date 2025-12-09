.PHONY: install dev test lint typecheck format clean run-api run-dashboard benchmark help

# Default target
.DEFAULT_GOAL := help

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pip install ruff mypy types-requests

install-dev: dev

# Llama setup
setup-llama:
	brew install ollama || echo "Ollama already installed or not on macOS"
	ollama pull llama3.1:8b

# Testing
test:
	python -m pytest tests/ -v --cov=rewardhackwatch --cov-report=term-missing

test-fast:
	python -m pytest tests/ -v -x --tb=short

test-ci:
	python -m pytest tests/ -v --cov=rewardhackwatch --cov-report=xml

# Linting & Formatting
lint:
	ruff check .

lint-fix:
	ruff check --fix .

typecheck:
	mypy rewardhackwatch --ignore-missing-imports

format:
	ruff format .

format-check:
	ruff format --check .

check: lint typecheck test
	@echo "All checks passed!"

# Running
api:
	uvicorn rewardhackwatch.api.main:app --reload --host 0.0.0.0 --port 8000

run-api: api

dashboard:
	streamlit run rewardhackwatch/dashboard/app.py --server.port 8501

run-dashboard: dashboard

cli:
	python -m rewardhackwatch.cli

# Benchmarks - RHW-Bench
benchmark:
	python -m rewardhackwatch.rhw_bench.benchmark_runner

experiment-benchmark:
	@echo "Running RHW-Bench experiment..."
	@mkdir -p results
	python -m rewardhackwatch.rhw_bench.benchmark_runner > results/benchmark_$$(date +%Y%m%d_%H%M%S).txt 2>&1
	@echo "Results saved to results/"
	@tail -50 results/benchmark_*.txt | head -50

benchmark-quick:
	python -m rewardhackwatch.rhw_bench.benchmark_runner rewardhackwatch/rhw_bench/test_cases

# Experiments
experiment-ablation:
	@echo "Running ablation study..."
	@mkdir -p results
	python -m rewardhackwatch.experiments.run_experiments
	@echo "Results saved to results/ablation_study.md"

experiment-train:
	@echo "Training ML models..."
	@mkdir -p models
	python -m rewardhackwatch.training.train_detector
	@echo "Models saved to models/"

experiment-full: experiment-train experiment-ablation experiment-benchmark
	@echo "All experiments complete!"
	@echo "Results:"
	@ls -la results/

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Help
help:
	@echo "RewardHackWatch - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install         - Install package"
	@echo "  make dev             - Install with dev dependencies"
	@echo "  make setup-llama     - Download Llama model"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run tests with coverage"
	@echo "  make test-fast       - Run tests (fast, stop on first failure)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            - Run ruff linter"
	@echo "  make typecheck       - Run mypy type checker"
	@echo "  make format          - Format code with ruff"
	@echo "  make check           - Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Benchmarks (RHW-Bench):"
	@echo "  make benchmark            - Run full RHW-Bench"
	@echo "  make experiment-benchmark - Run benchmark and save results to file"
	@echo ""
	@echo "Experiments:"
	@echo "  make experiment-train     - Train ML models"
	@echo "  make experiment-ablation  - Run ablation study"
	@echo "  make experiment-full      - Run all experiments"
	@echo ""
	@echo "Running:"
	@echo "  make api             - Start FastAPI server (port 8000)"
	@echo "  make dashboard       - Start Streamlit dashboard (port 8501)"
	@echo "  make cli             - Run CLI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-up       - Start services"
	@echo "  make docker-down     - Stop services"
	@echo ""
	@echo "Other:"
	@echo "  make clean           - Clean build artifacts"
