import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.pipeline.prompt_templates import sql_prompt, answer_prompt, AGENT_SYSTEM_PROMPT
from src.tools import get_schema_and_samples, execute_sql, init_tools


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


def _extract_sql(ai_message) -> dict:
    """Takes the raw AIMessage from the LLM, extracts cleaned SQL text plus token usage."""
    sql = ai_message.content.strip()
    if sql.startswith("```"):
        sql = sql.strip("`").replace("sql\n", "", 1).strip()
    usage = getattr(ai_message, "usage_metadata", None) or {}
    return {
        "sql": sql,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def _extract_answer(ai_message) -> dict:
    """Takes the raw AIMessage from the LLM, extracts the answer text plus token usage."""
    usage = getattr(ai_message, "usage_metadata", None) or {}
    return {
        "answer": ai_message.content,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


# ─────────────────────────────────────────────────────────
# LEGACY: two-chain pipeline (kept for reference / fallback)
# ─────────────────────────────────────────────────────────

def build_nl_to_sql_chain(metadata: dict, model: str = "openai/gpt-oss-120b"):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    llm = ChatGroq(model=model, temperature=0, api_key=api_key)
    chain = sql_prompt | llm | RunnableLambda(_extract_sql)

    schema_summary = _schema_json(metadata)
    lov_context = _lov_json(metadata)

    def generate(question: str, error_feedback: str = None, callbacks: list = None) -> dict:
        if error_feedback:
            question = f"{question}\n\nPrevious attempt failed with error: {error_feedback}\nPlease correct the SQL."
        config = {"callbacks": callbacks} if callbacks else {}
        return chain.invoke({
            "schema_summary": schema_summary,
            "lov_context": lov_context,
            "question": question,
        }, config=config)

    return generate


def build_answer_chain(model: str = "openai/gpt-oss-120b"):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    llm = ChatGroq(model=model, temperature=0, api_key=api_key)
    chain = answer_prompt | llm | RunnableLambda(_extract_answer)

    def answer(question: str, columns: list, rows: list, callbacks: list = None) -> dict:
        truncated_rows = rows[:20]
        rows_text = "\n".join(str(r) for r in truncated_rows)
        if len(rows) > 20:
            rows_text += f"\n... and {len(rows) - 20} more rows (total {len(rows)} rows)"
        config = {"callbacks": callbacks} if callbacks else {}
        return chain.invoke({
            "question": question,
            "columns": columns,
            "rows": rows_text,
        }, config=config)

    return answer


# ─────────────────────────────────────────────────────────
# Single agent, tool-calling, LangChain (AgentExecutor) — manual chat history
# ─────────────────────────────────────────────────────────

def build_agent(metadata: dict, engine, model: str = "openai/gpt-oss-120b"):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    init_tools(metadata=metadata, engine=engine, allowed_tables=set(metadata.keys()))

    llm = ChatGroq(model=model, temperature=0, api_key=api_key)
    tools = [get_schema_and_samples, execute_sql]

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor


# ─────────────────────────────────────────────────────────
# Single agent with LangGraph-managed memory (checkpointer)
# ─────────────────────────────────────────────────────────

def build_agent_with_memory(metadata: dict, engine, model: str = "openai/gpt-oss-120b"):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    init_tools(metadata=metadata, engine=engine, allowed_tables=set(metadata.keys()))

    llm = ChatGroq(model=model, temperature=0, api_key=api_key)
    tools = [get_schema_and_samples, execute_sql]

    checkpointer = MemorySaver()

    agent = create_react_agent(
    llm,
    tools,
    state_modifier=AGENT_SYSTEM_PROMPT,
    checkpointer=checkpointer,
)
    return agent