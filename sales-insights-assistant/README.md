# sales-insights-assistant

Ask questions about the Northwind sales database in plain English, get back the SQL and an answer. Built with LangChain (LCEL) + Groq, with a bunch of guardrails so it can't run anything dangerous or leak personal data.

## How it works

1. **Metadata extraction** (`src/db/metadata_extractor.py`) - the LLM knows nothing about Northwind, so this pulls the schema + some sample values from Postgres and feeds them into the prompt
2. **SQL generation** (`src/pipeline/main.py`) - first LLM call turns the question into SQL
3. **Guardrails** (`src/guardrails.py`)
   - input check - blocks injection / off-topic stuff
   - SQL validator - parses the query with `sqlglot` (AST), only allows read-only SELECTs
   - result sanitizer - redacts PII columns before anything goes back to the LLM
4. **Answer generation** - second LLM call turns the rows into a readable answer
5. **Logging + tracing** - every prompt logged as JSON (`src/logger.py`), plus Langfuse for tokens and latency
6. **UI** - Streamlit chat app (`src/app.py`)

Architecture diagram is in `docs/`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Groq key (+ Langfuse keys if you want tracing)
```

Load Northwind into Postgres (scripts are in `src/db/scripts/`):

```bash
createdb -U postgres northwind
psql -U postgres -d northwind -f src/db/scripts/northwind_ddl.sql
psql -U postgres -d northwind -f src/db/scripts/northwind_data.sql
```

## Run

```bash
streamlit run src/app.py
```

Eval harness (15 test questions):
```bash
python tests/eval_harness.py
```
