"""
SQL-side safety checks used by mcp_server.py's execute_sql tool.

Trimmed down from the original NL-to-SQL project's src/guardrails.py:
this multi-agent system has its own prompt-injection guardrail (the root
guardrails.py, run as the graph's first node), so the old LLM-based
InputGuardrail here was dead code and has been removed. Only the three
classes mcp_server.py actually imports remain:

  - SQLValidator:    is this even a SELECT, and does it only touch known tables?
  - OutputGuardrail:  same checks again post-generation, plus enforces a row LIMIT
  - ResultSanitizer:  strips sensitive columns (email/phone/fax) from results
"""

import sqlglot
from sqlglot import exp


ALLOWED_STATEMENT_TYPES = (exp.Select,)


class SQLValidator:
    def __init__(self, allowed_tables: set):
        self.allowed_tables = {t.lower() for t in allowed_tables}

    def validate(self, sql: str) -> tuple[bool, str]:
        try:
            parsed = sqlglot.parse_one(sql, read="sqlite")
        except Exception as e:
            return False, f"SQL could not be parsed: {e}"

        if not isinstance(parsed, ALLOWED_STATEMENT_TYPES):
            return False, f"Only SELECT statements are allowed, got: {type(parsed).__name__}"

        referenced_tables = {
            table.name.lower() for table in parsed.find_all(exp.Table)
        }

        unknown_tables = referenced_tables - self.allowed_tables
        if unknown_tables:
            return False, f"Query references unknown tables: {unknown_tables}"

        return True, "OK"


DEFAULT_LIMIT = 100


class OutputGuardrail:
    def __init__(self, permitted_tables: set):
        self.permitted_tables = {t.lower() for t in permitted_tables}

    def check(self, sql: str) -> tuple[bool, str, str]:
        try:
            parsed = sqlglot.parse_one(sql, read="sqlite")
        except Exception as e:
            return False, f"SQL could not be parsed: {e}", sql

        if not isinstance(parsed, exp.Select):
            return False, "Only read-only SELECT statements are permitted", sql

        joined_tables = {
            table.name.lower() for table in parsed.find_all(exp.Table)
        }
        not_permitted = joined_tables - self.permitted_tables
        if not_permitted:
            return False, f"Query touches non-permitted tables: {not_permitted}", sql

        if not parsed.args.get("limit"):
            parsed = parsed.limit(DEFAULT_LIMIT)
            sql = parsed.sql(dialect="sqlite")

        return True, "OK", sql


SENSITIVE_COLUMNS = {
    "phone", "fax", "email",
}


class ResultSanitizer:
    def sanitize(self, columns: list, rows: list) -> tuple[list, list, str]:
        """
        Returns (filtered_columns, filtered_rows, message).
        message is a note about what happened (e.g. columns hidden, empty result explanation).
        """
        message = ""

        sensitive_indexes = [
            i for i, col in enumerate(columns) if col.lower() in SENSITIVE_COLUMNS
        ]

        if sensitive_indexes:
            hidden_names = [columns[i] for i in sensitive_indexes]
            columns = [c for i, c in enumerate(columns) if i not in sensitive_indexes]
            rows = [
                tuple(v for i, v in enumerate(row) if i not in sensitive_indexes)
                for row in rows
            ]
            message = f"Note: the following sensitive columns were hidden from results: {hidden_names}"

        if not rows:
            message = (
                "No results were found. This could mean there is genuinely no matching data, "
                "or the generated query may not have captured your intent correctly."
            )

        return columns, rows, message
