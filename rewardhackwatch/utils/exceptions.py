"""Custom exceptions for RewardHackWatch."""


class RewardHackWatchError(Exception):
    """Base exception for all RewardHackWatch errors."""

    pass


class ConfigurationError(RewardHackWatchError):
    """Error in configuration."""

    pass


class DetectionError(RewardHackWatchError):
    """Error during detection."""

    pass


class AnalysisError(RewardHackWatchError):
    """Error during analysis."""

    pass


class TrajectoryError(RewardHackWatchError):
    """Error with trajectory data."""

    pass


class InvalidTrajectoryError(TrajectoryError):
    """Trajectory data is invalid or malformed."""

    pass


class TrajectoryNotFoundError(TrajectoryError):
    """Trajectory not found."""

    pass


class ModelError(RewardHackWatchError):
    """Error with ML model."""

    pass


class ModelNotLoadedError(ModelError):
    """Model not loaded or unavailable."""

    pass


class ModelInferenceError(ModelError):
    """Error during model inference."""

    pass


class APIError(RewardHackWatchError):
    """Error in API operations."""

    pass


class RateLimitError(APIError):
    """Rate limit exceeded."""

    pass


class AuthenticationError(APIError):
    """Authentication failed."""

    pass


class CacheError(RewardHackWatchError):
    """Error with caching operations."""

    pass


class CacheMissError(CacheError):
    """Cache miss error."""

    pass


class IntegrationError(RewardHackWatchError):
    """Error with external integration."""

    pass


class WebhookError(IntegrationError):
    """Error sending webhook."""

    pass


class NotificationError(IntegrationError):
    """Error sending notification."""

    pass
