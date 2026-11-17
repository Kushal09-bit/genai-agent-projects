# genai-agent-projects

LLM agent projects I built during my GenAI engineer internship (2026). Mostly agents, RAG, text-to-SQL, guardrails and evals - each folder is its own standalone project with its own README and requirements.

| Project | What it is | Stack |
|---|---|---|
| [agentic-rag-clinical-trials](agentic-rag-clinical-trials) | ReAct agent that answers questions across the AACT clinical trials DB (SQL) and biomedical papers (RAG), tools served over MCP, split into 3 docker services | LangGraph, MCP, ChromaDB, BM25 + dense hybrid retrieval, FastAPI, Streamlit, Ragas, Langfuse |
| [biomedical-research-assistant](biomedical-research-assistant) | RAG chatbot over PubMed papers - OCR ingestion, cross-encoder reranking, citations, guardrails | LangGraph, MCP, ChromaDB, Groq, Streamlit, Ragas, Docker |
| [text-to-sql-chatbot](text-to-sql-chatbot) | Text-to-SQL chatbot on the Chinook music DB, single tool-calling agent with memory | LangChain, LangGraph, Groq, PostgreSQL, Ragas, Langfuse |
| [sales-insights-assistant](sales-insights-assistant) | NL-to-SQL assistant on the Northwind DB with SQL guardrails and PII redaction | LangChain (LCEL), Groq, sqlglot, PostgreSQL, Streamlit, Langfuse |

## Running any of them

```bash
cd <project-folder>
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # agentic-rag has its requirements in requirements/
cp .env.example .env              # then fill in your own keys
```

Then follow that project's README.

Note: the research papers used for the RAG projects aren't included here (copyright) - drop any PubMed PDFs into the docs folder mentioned in each README and run the ingestion step.
