import json
from sqlalchemy import text
from langchain_core.tools import tool

from src.guardrails import SQLValidator, OutputGuardrail, ResultSanitizer

# Module-level state, set once via init_tools() at app startup
_metadata_store = {}
_engine_store = {}
_validator = None
_output_guardrail = None
_result_sanitizer = ResultSanitizer()


def init_tools(metadata: dict, engine, allowed_tables: set):
    """Call once at startup (in load_pipeline) before the agent is built."""
    global _validator, _output_guardrail
    _metadata_store["schema"] = metadata
    _engine_store["engine"] = engine
    _validator = SQLValidator(allowed_tables=allowed_tables)
    _output_guardrail = OutputGuardrail(permitted_tables=allowed_tables)


def _schema_json(metadata: dict) -> str:
    return json.dumps(
        {t: [{"name": c.name, "type": c.type} for c in m.columns] for t, m in metadata.items()},
        default=str,
    )


def _lov_json(metadata: dict) -> str:
    return json.dumps(
        {f"{t}.{c.name}": c.sample_values for t, m in metadata.items() for c in m.columns if c.sample_values},
        default=str,
    )


@tool
def get_schema_and_samples() -> str:
    """Return the Chinook database schema (tables, columns, types) and example values for key columns.
    Always call this before writing any SQL query."""
    metadata = _metadata_store["schema"]
    schema = _schema_json(metadata)
    samples = _lov_json(metadata)
    return f"Schema:\n{schema}\n\nExample values for key columns:\n{samples}"


@tool
def execute_sql(sql: str) -> str:
    """Validate and execute a read-only SQL SELECT query against the Chinook database.
    Returns a JSON string with columns, rows, row_count, and any notes (e.g. hidden sensitive columns).
    If the query is invalid or unsafe, returns an INVALID message with the reason instead — read it,
    correct the SQL, and call this tool again."""
    valid, reason = _validator.validate(sql)
    if not valid:
        return f"INVALID: {reason}"

    valid, reason, sql = _output_guardrail.check(sql)
    if not valid:
        return f"INVALID: {reason}"

    engine = _engine_store["engine"]
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
    except Exception as e:
        return f"EXECUTION_ERROR: {e}"

    columns, rows, note = _result_sanitizer.sanitize(columns, rows)

    payload = {
        "sql": sql,
        "columns": columns,
        "rows": [list(r) for r in rows[:20]],
        "row_count": len(rows),
        "note": note,
    }
    return json.dumps(payload, default=str)