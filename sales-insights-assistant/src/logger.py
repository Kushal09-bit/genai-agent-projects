import re
import uuid
import json
import os
from datetime import datetime, timezone

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "pipeline_logs.jsonl")

EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"


def new_trace_id() -> str:
    """One unique ID per user question, used to tie all its log entries together."""
    return str(uuid.uuid4())


def new_session_id() -> str:
    """One unique ID per user session (conversation), persists across multiple questions."""
    return str(uuid.uuid4())


def _sanitize(value):
    if not isinstance(value, str):
        return value
    value = re.sub(EMAIL_PATTERN, "[REDACTED_EMAIL]", value)
    value = re.sub(PHONE_PATTERN, "[REDACTED_PHONE]", value)
    return value


def log_event(component: str, event: str, trace_id: str, session_id: str = None, **payload):
    """
    Required structured log call for every stage of the pipeline.

    component  : which part of the system produced this log (e.g. 'input_guardrail')
    event      : the event type (e.g. 'input_received', 'sql_generated')
    trace_id   : ties every entry for one user question together
    session_id : ties every entry for one user's full conversation together
    payload    : any extra data, automatically sanitized for PII patterns
    """
    clean_payload = {k: _sanitize(v) for k, v in payload.items()}

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "info",
        "component": component,
        "event": event,
        "trace_id": trace_id,
        "session_id": session_id,
        **clean_payload,
    }

    log_line = json.dumps(entry)

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

    print(log_line)