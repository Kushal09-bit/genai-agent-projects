import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from state import AgentState


def aggregator_node(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
    route = state["route"]
    parts = []

    # Only include an answer if the CURRENT turn's route actually ran that
    # sub-agent — not just because the field happens to be non-empty, since
    # a previous turn's leftover sql_answer/rag_answer would otherwise leak
    # into a turn that never re-ran that sub-agent.
    if route in ("sql", "both") and state.get("sql_answer"):
        answer = state["sql_answer"]
        sql_query = state.get("sql_query")
        if sql_query:
            answer = f"{answer}\n\nQuery used: `{sql_query}`"
        parts.append(answer)

    if route in ("rag", "both") and state.get("rag_answer"):
        answer = state["rag_answer"]
        citations = state.get("rag_citations") or []
        if citations:
            unique_citations = list(dict.fromkeys(citations))  # de-dupe, keep order
            answer = f"{answer}\n\nSources: {', '.join(unique_citations)}"
        parts.append(answer)

    final_answer = "\n\n".join(parts) if parts else "I don't have an answer for that."

    # Write to the long-term Store, keyed by user_id (NOT thread_id) — this
    # is what makes it survive into a brand new conversation. Falls back to
    # "anonymous" if no user_id was provided in config, so this never
    # crashes a run that doesn't care about long-term memory.
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    namespace = (user_id, "qa_history")
    store.put(namespace, str(uuid.uuid4()), {
        "question": state["question"],
        "answer": final_answer,
        "route": route,
    })

    # Append this turn to the THREAD-scoped history too (Day 4's feature,
    # unchanged) — this is separate from the store write above: history
    # is per-conversation, the store write above is per-user-forever.
    return {
        "final_answer": final_answer,
        "history": [{"question": state["question"], "answer": final_answer}],
    }


def off_topic_node(state: AgentState) -> dict:
    """Runs instead of any sub-agent when the supervisor classifies the
    question as unrelated to either data source."""
    final_answer = (
        "I can help with questions about the Chinook music-store data "
        "or the biomedical research literature. Could you rephrase your "
        "question to relate to one of those?"
    )
    return {
        "final_answer": final_answer,
        "history": [{"question": state["question"], "answer": final_answer}],
    }