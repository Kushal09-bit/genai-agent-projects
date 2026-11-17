---
name: sql-skill
description: Safely writing and running SQL against the AACT clinical trials database - check real schema and values before querying, validate before executing.
allowed-tools: [schema_tool, values_tool, sql_gen_tool, sql_exec_tool]
---

## Querying AACT (SQL skill)

- Call `schema_tool` before writing SQL against a table you haven't already
  checked this conversation - don't guess column names.
- Call `values_tool` before filtering on a categorical column (like
  `overall_status` or `phase`) so you're matching real values, not guesses.
- `nct_id` is the join key across almost every AACT table.
- Always run your SQL through `sql_gen_tool` before `sql_exec_tool` - it'll
  tell you exactly what's wrong if something's not allowed.
- Keep result sets small (a LIMIT of 20-50 is usually plenty) - AACT is a
  shared public resource, not just yours.
