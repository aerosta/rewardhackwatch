# RewardHackWatch Dockerfile
# Python 3.12 base with all dependencies

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml setup.py ./
COPY rewardhackwatch/__init__.py rewardhackwatch/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Create directory for SQLite database
RUN mkdir -p /data

# Expose ports
# API port
EXPOSE 8000
# Dashboard port
EXPOSE 8501

# Default command runs the API
CMD ["uvicorn", "rewardhackwatch.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
