"""JSONL batch loader with automatic schema detection.

Supports multiple JSONL formats out of the box:

* **claude_session** — Claude Code tool-use sessions (``message.id``,
  file-history-snapshot blocks, etc.)
* **type_content** — ``{"type": "human"|"assistant", "content": ...}``
* **role_content** — ``{"role": "user"|"assistant", "content": ...}``
* **sender_message** — ``{"sender": "human"|"bot", "message": ...}``
* **nested_data** — ``{"data": {"role": ..., "content": ...}}``
* **trajectory** — RewardHackWatch trajectory format with ``cot_traces``
  / ``code_outputs`` / ``steps``.
* **unknown** — fallback that tries to extract the first string value.

Usage::

    from rewardhackwatch.eval import load_jsonl, load_jsonl_dir

    session = load_jsonl("session.jsonl")
    sessions = load_jsonl_dir("data/sessions/")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .models import ParsedSession, Role, Turn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role normalisation
# ---------------------------------------------------------------------------

_ROLE_MAP: dict[str, Role] = {
    "human": Role.HUMAN,
    "user": Role.HUMAN,
    "customer": Role.HUMAN,
    "assistant": Role.ASSISTANT,
    "bot": Role.ASSISTANT,
    "ai": Role.ASSISTANT,
    "model": Role.ASSISTANT,
    "system": Role.SYSTEM,
    "tool_use": Role.TOOL_USE,
    "tool_result": Role.TOOL_RESULT,
    "tool": Role.TOOL_RESULT,
}


def _normalise_role(raw: str) -> Role:
    return _ROLE_MAP.get(raw.strip().lower(), Role.UNKNOWN)


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(value: Any) -> str:
    """Recursively flatten *value* into a plain-text string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", "") or item.get("content", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    if isinstance(value, dict):
        return str(value.get("text", "") or value.get("content", "") or "")
    return str(value) if value else ""


def _extract_tool_calls(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull tool-call metadata out of a raw JSON line."""
    calls: list[dict[str, Any]] = []
    content = obj.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                })
    return calls


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

_SCHEMA_NAMES = (
    "claude_session",
    "type_content",
    "role_content",
    "sender_message",
    "nested_data",
    "trajectory",
    "unknown",
)


def detect_schema(obj: dict[str, Any]) -> str:
    """Return the most likely schema name for a parsed JSON object."""
    # Claude Code session format
    if "message" in obj and isinstance(obj.get("message"), dict):
        msg = obj["message"]
        if "id" in msg or "uuid" in obj or "sessionId" in obj:
            return "claude_session"

    # RewardHackWatch trajectory format
    if "trajectory" in obj or "cot_traces" in obj or "code_outputs" in obj:
        return "trajectory"

    # {"type": "human", "content": ...}
    if "type" in obj and "content" in obj:
        t = str(obj["type"]).lower()
        if t in ("human", "assistant", "user", "system"):
            return "type_content"

    # {"role": "user", "content": ...}
    if "role" in obj and "content" in obj:
        return "role_content"

    # {"sender": "human", "message": "..."}
    if "sender" in obj and "message" in obj:
        return "sender_message"

    # {"data": {"role": ..., "content": ...}}
    if "data" in obj and isinstance(obj["data"], dict):
        inner = obj["data"]
        if "role" in inner or "content" in inner:
            return "nested_data"

    return "unknown"


# ---------------------------------------------------------------------------
# Per-schema turn extraction
# ---------------------------------------------------------------------------

def _extract_turn(obj: dict[str, Any], schema: str, index: int) -> Turn | None:
    """Convert a single JSON object into a Turn given its *schema*."""
    if schema == "claude_session":
        msg = obj.get("message", {})
        role_str = msg.get("role", obj.get("type", "unknown"))
        content = _extract_text(msg.get("content", ""))
        ts = obj.get("timestamp") or obj.get("createdAt")
        return Turn(
            index=index,
            role=_normalise_role(role_str),
            content=content,
            timestamp=str(ts) if ts else None,
            tool_calls=_extract_tool_calls(msg),
            raw=obj,
        )

    if schema == "trajectory":
        # Convert trajectory into a single pseudo-turn for downstream use
        inner = obj.get("trajectory", obj)
        parts: list[str] = []
        for trace in inner.get("cot_traces", []):
            parts.append(f"[CoT] {trace}")
        for code in inner.get("code_outputs", []):
            parts.append(f"[Code] {code}")
        for step in inner.get("steps", []):
            action = step.get("action", "")
            reasoning = step.get("reasoning", "")
            if action:
                parts.append(f"[Step] {action}")
            if reasoning:
                parts.append(f"[Reasoning] {reasoning}")
        return Turn(
            index=index,
            role=Role.ASSISTANT,
            content="\n".join(parts) if parts else json.dumps(inner)[:500],
            raw=obj,
        )

    if schema == "type_content":
        return Turn(
            index=index,
            role=_normalise_role(str(obj.get("type", "unknown"))),
            content=_extract_text(obj.get("content", "")),
            timestamp=obj.get("timestamp"),
            tool_calls=_extract_tool_calls(obj),
            raw=obj,
        )

    if schema == "role_content":
        return Turn(
            index=index,
            role=_normalise_role(str(obj.get("role", "unknown"))),
            content=_extract_text(obj.get("content", "")),
            timestamp=obj.get("timestamp"),
            tool_calls=_extract_tool_calls(obj),
            raw=obj,
        )

    if schema == "sender_message":
        return Turn(
            index=index,
            role=_normalise_role(str(obj.get("sender", "unknown"))),
            content=_extract_text(obj.get("message", "")),
            timestamp=obj.get("timestamp"),
            raw=obj,
        )

    if schema == "nested_data":
        inner = obj.get("data", {})
        return Turn(
            index=index,
            role=_normalise_role(str(inner.get("role", "unknown"))),
            content=_extract_text(inner.get("content", "")),
            timestamp=inner.get("timestamp") or obj.get("timestamp"),
            raw=obj,
        )

    # unknown — grab first string value
    for v in obj.values():
        if isinstance(v, str) and len(v) > 1:
            return Turn(index=index, role=Role.UNKNOWN, content=v, raw=obj)
    return None


# ---------------------------------------------------------------------------
# Claude-session message merging
# ---------------------------------------------------------------------------

def _merge_claude_messages(turns: list[Turn]) -> list[Turn]:
    """Merge fragmented Claude Code assistant messages by message id."""
    merged: list[Turn] = []
    pending: dict[str, Turn] = {}  # message_id -> accumulated turn

    for turn in turns:
        msg_id = (turn.raw.get("message", {}) or {}).get("id", "")
        if not msg_id:
            # No message id — flush pending and append
            merged.extend(pending.values())
            pending.clear()
            merged.append(turn)
            continue

        if msg_id in pending:
            prev = pending[msg_id]
            prev.content = prev.content + "\n" + turn.content if turn.content else prev.content
            prev.tool_calls.extend(turn.tool_calls)
            prev.word_count = len(prev.content.split())
        else:
            pending[msg_id] = turn

    merged.extend(pending.values())
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_jsonl(
    path: str | Path,
    *,
    schema: str | None = None,
    merge_messages: bool = True,
) -> ParsedSession:
    """Parse a JSONL file into a :class:`ParsedSession`.

    Args:
        path: Path to a ``.jsonl`` file.
        schema: Force a specific schema name instead of auto-detecting.
        merge_messages: Merge fragmented Claude Code messages (default True).

    Returns:
        A :class:`ParsedSession` with extracted turns.
    """
    path = Path(path)
    turns: list[Turn] = []
    detected_schema = schema or "unknown"
    line_count = 0
    error_count = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                error_count += 1
                continue

            if not isinstance(obj, dict):
                continue

            # Detect schema from first valid object
            if detected_schema == "unknown" and schema is None:
                detected_schema = detect_schema(obj)

            turn = _extract_turn(obj, detected_schema, index=len(turns))
            if turn is not None:
                turns.append(turn)

    if merge_messages and detected_schema == "claude_session":
        turns = _merge_claude_messages(turns)

    # Re-index after merging
    for i, turn in enumerate(turns):
        turn.index = i

    logger.info(
        "Parsed %s: %d lines, %d turns, schema=%s, errors=%d",
        path.name, line_count, len(turns), detected_schema, error_count,
    )

    return ParsedSession(
        turns=turns,
        schema=detected_schema,
        source_path=str(path),
        metadata={
            "line_count": line_count,
            "error_count": error_count,
        },
    )


def load_jsonl_dir(
    directory: str | Path,
    *,
    pattern: str = "*.jsonl",
    schema: str | None = None,
) -> list[ParsedSession]:
    """Load all JSONL files from a directory.

    Args:
        directory: Directory to scan.
        pattern: Glob pattern for matching files.
        schema: Force a specific schema (auto-detect if None).

    Returns:
        List of parsed sessions, one per file.
    """
    directory = Path(directory)
    sessions: list[ParsedSession] = []

    for path in sorted(directory.glob(pattern)):
        if path.is_file():
            try:
                session = load_jsonl(path, schema=schema)
                sessions.append(session)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

    logger.info("Loaded %d sessions from %s", len(sessions), directory)
    return sessions
