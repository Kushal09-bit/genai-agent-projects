# Music Store Insights Assistant

This project started as a clone of an existing **Sales Insights Assistant** project (built on the Northwind sales database) and was improved and adapted into a Text-to-SQL chatbot for the **Chinook music store database**.

## What Changed From the Original
- Migrated from hardcoded sequential LLM calls to a single **LangChain tool-calling agent**
- Added **conversation memory** via LangGraph's checkpointer, so the assistant can handle follow-up questions
- Added a **Ragas-based evaluation suite** (faithfulness, answer relevancy, answer correctness)
- Switched the underlying database from **Northwind (sales)** to **Chinook (music store)** — updated schema handling, system prompt, sensitive-column guardrails, and UI text accordingly
- Kept the original safety guardrails (input classification, SQL validation, output sanitization) and Langfuse tracing intact

## Features
- Single LangChain agent with tool-calling (schema lookup + SQL execution)
- Conversation memory across follow-up questions
- Input/output guardrails and SQL validation
- Langfuse tracing for observability
- Ragas evaluation (faithfulness, answer relevancy, answer correctness)

## Tech Stack
- LangChain + LangGraph
- Groq (LLM inference)
- PostgreSQL (Chinook database)
- Streamlit (UI)
- Ragas (evaluation)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set up `.env` with your Groq API key
3. Ensure the Chinook database is running in Postgres
4. Run the app: `streamlit run src/app.py`

## Evaluation
Run the Ragas evaluation suite:
```bash
python3 evaluation/run_eval.py
```