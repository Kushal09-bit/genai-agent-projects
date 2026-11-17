"""
day4: everything about talking to AACT safely - connection, caching, sql
validation, and a handful of typed query helpers. src/mcp_server.py is the
only thing that calls into this; it's the AACT-specific data-access-and-
safety layer, kept separate from the MCP protocol wiring on purpose - this
is what actually gets unit-tested directly, no MCP/stdio needed.

AACT is a shared resource other people are also hitting, so this is fairly
paranoid on purpose:
  - the db session itself is opened read-only (set_session(readonly=True)),
    so even if validate_sql below has a gap, postgres itself refuses any
    write
  - a statement_timeout so one bad query can't hang forever
  - every result gets cached in-process (schema/values/query results don't
    change mid-session) so the agent re-asking something doesn't mean
    another round trip
  - table/column names that get interpolated into SQL are checked against
    information_schema first and composed with psycopg2.sql.Identifier,
    never f-string'd straight in
"""

import hashlib
import re
from collections import OrderedDict
from functools import wraps

import psycopg2
import psycopg2.extras
from psycopg2 import sql as pgsql

from src.config import (
    AACT_DB_HOST,
    AACT_DB_NAME,
    AACT_DB_PASSWORD,
    AACT_DB_PORT,
    AACT_DB_USER,
    AACT_MAX_ROWS,
    AACT_SCHEMA,
    AACT_STATEMENT_TIMEOUT_MS,
)

# ---------- connection ----------

_conn = None


def get_connection():
    """lazily opens (or reopens, if it died) a read-only connection to AACT."""
    global _conn
    if _conn is not None and not _conn.closed:
        return _conn

    if not all([AACT_DB_HOST, AACT_DB_NAME, AACT_DB_USER, AACT_DB_PASSWORD]):
        raise RuntimeError(
            "AACT_DB_HOST/NAME/USER/PASSWORD aren't all set in .env - "
            "get credentials at https://aact.ctti-clinicaltrials.org"
        )

    _conn = psycopg2.connect(
        host=AACT_DB_HOST,
        port=AACT_DB_PORT,
        dbname=AACT_DB_NAME,
        user=AACT_DB_USER,
        password=AACT_DB_PASSWORD,
        connect_timeout=10,
    )
    _conn.set_session(readonly=True, autocommit=True)
    with _conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {AACT_STATEMENT_TIMEOUT_MS}")
    return _conn


def run_query(sql, params=None):
    """runs a SELECT and returns rows as a list of plain dicts."""
    conn = get_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


# ---------- tiny in-process cache ----------
# AACT reference data doesn't change mid-session, so caching schema/values/
# query lookups is free correctness-wise and saves a lot of round trips to
# a database we don't own. plain LRU dict, no TTL needed for this lifetime.

_CACHE_MAX_ENTRIES = 256


def cached(fn):
    store = OrderedDict()

    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = hashlib.sha256(repr((args, sorted(kwargs.items()))).encode()).hexdigest()
        if key in store:
            store.move_to_end(key)
            return store[key]
        result = fn(*args, **kwargs)
        store[key] = result
        if len(store) > _CACHE_MAX_ENTRIES:
            store.popitem(last=False)
        return result

    wrapper.cache_clear = store.clear
    return wrapper


# ---------- sql safety ----------

_DISALLOWED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|EXEC|CALL|MERGE|COPY)\b",
    re.IGNORECASE,
)
_STARTS_WITH_SELECT = re.compile(r"^\s*(WITH\b.*?)?SELECT\b", re.IGNORECASE | re.DOTALL)
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)


def validate_sql(sql: str) -> str:
    """raises ValueError with a human-readable reason if sql isn't a safe
    single read-only SELECT. otherwise returns it, LIMIT-capped if it
    didn't already have one."""
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    if ";" in stripped:
        raise ValueError("one statement at a time - drop the extra ';' and whatever follows it")
    if not _STARTS_WITH_SELECT.match(stripped):
        raise ValueError("only SELECT (optionally with a WITH/CTE) is allowed against AACT")
    bad = _DISALLOWED.search(stripped)
    if bad:
        raise ValueError(f"'{bad.group(0)}' isn't allowed - AACT is a shared read-only resource")
    if not _HAS_LIMIT.search(stripped):
        stripped = f"{stripped} LIMIT {AACT_MAX_ROWS}"
    return stripped


def _assert_real_column(table_name: str, column_name: str) -> None:
    """table/column names can't go through a %s placeholder (those are only
    for values), so before building SQL with them we check they're actually
    real ctgov columns - closes off the injection angle."""
    rows = run_query(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """,
        (AACT_SCHEMA, table_name, column_name),
    )
    if not rows:
        raise ValueError(
            f"'{table_name}.{column_name}' isn't a real column in the {AACT_SCHEMA} schema - "
            "double check with list_tables/list_columns"
        )


# ---------- typed query helpers ----------

def list_tables() -> list[dict]:
    return run_query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s ORDER BY table_name",
        (AACT_SCHEMA,),
    )


def list_columns(table_name: str) -> list[dict]:
    return run_query(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (AACT_SCHEMA, table_name),
    )


def list_values(table_name: str, column_name: str, limit: int = 20) -> list:
    """raises ValueError (via _assert_real_column) if column_name isn't real."""
    _assert_real_column(table_name, column_name)
    query = pgsql.SQL(
        "SELECT {col} AS value, COUNT(*) AS n FROM {schema}.{table} "
        "WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY n DESC LIMIT %s"
    ).format(
        col=pgsql.Identifier(column_name),
        schema=pgsql.Identifier(AACT_SCHEMA),
        table=pgsql.Identifier(table_name),
    )
    rows = run_query(query, (min(limit, 100),))
    return [r["value"] for r in rows]


def get_study(nct_id: str) -> dict | None:
    rows = run_query(
        f"""
        SELECT nct_id, brief_title, official_title, overall_status, phase
        FROM {AACT_SCHEMA}.studies
        WHERE nct_id = %s
        """,
        (nct_id,),
    )
    return rows[0] if rows else None


def get_conditions(nct_id: str) -> list[str]:
    rows = run_query(f"SELECT name FROM {AACT_SCHEMA}.conditions WHERE nct_id = %s", (nct_id,))
    return [r["name"] for r in rows]
