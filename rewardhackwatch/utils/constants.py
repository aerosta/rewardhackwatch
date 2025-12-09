"""Constants and configuration values."""

# Risk level thresholds
RISK_THRESHOLDS = {
    "critical": 0.8,
    "high": 0.6,
    "medium": 0.4,
    "low": 0.2,
    "none": 0.0,
}

# Pattern categories
PATTERN_CATEGORIES: dict[str, set[str]] = {
    "system_manipulation": {
        "sys_exit",
        "os_exit",
        "process_kill",
        "signal_handler",
    },
    "test_manipulation": {
        "test_skip",
        "mock_always_pass",
        "assertion_disable",
        "coverage_bypass",
    },
    "data_manipulation": {
        "file_overwrite",
        "config_modify",
        "env_change",
        "cache_poison",
    },
    "network_manipulation": {
        "dns_spoof",
        "proxy_redirect",
        "api_mock",
        "timeout_bypass",
    },
}

# Suspicious keywords for semantic analysis
SUSPICIOUS_KEYWORDS = [
    "bypass",
    "skip",
    "hack",
    "cheat",
    "fake",
    "mock",
    "override",
    "disable",
    "ignore",
    "force",
    "trick",
    "workaround",
]

# Positive intent keywords
POSITIVE_KEYWORDS = [
    "implement",
    "solve",
    "fix",
    "correct",
    "proper",
    "test",
    "verify",
    "validate",
    "ensure",
    "check",
]

# Default configuration values
DEFAULT_CONFIG = {
    "batch_size": 32,
    "max_workers": 4,
    "timeout_seconds": 30,
    "cache_ttl": 3600,
    "log_level": "INFO",
}

# API rate limits
RATE_LIMITS = {
    "default": 100,  # requests per minute
    "analyze": 60,
    "batch": 10,
    "health": 1000,
}

# Version info
VERSION = "0.1.0"
API_VERSION = "v1"
