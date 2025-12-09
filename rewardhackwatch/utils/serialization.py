"""Serialization utilities."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common Python types."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def to_json(obj: Any, pretty: bool = False) -> str:
    """Convert object to JSON string."""
    indent = 2 if pretty else None
    return json.dumps(obj, cls=EnhancedJSONEncoder, indent=indent)


def from_json(json_str: str) -> Any:
    """Parse JSON string."""
    return json.loads(json_str)


def safe_serialize(obj: Any) -> dict[str, Any]:
    """Safely serialize object to dict, handling errors."""
    try:
        return json.loads(to_json(obj))
    except (TypeError, ValueError):
        return {"error": "Could not serialize", "type": type(obj).__name__}
