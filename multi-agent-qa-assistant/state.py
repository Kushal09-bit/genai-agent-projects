"""
Shared state for the supervisor graph. Every node reads from and writes
back to this — think of it as a clipboard passed between the supervisor
and whichever sub-agent handles the question.
"""

from typing import Annotated, List, Literal, Optional, TypedDict


def _add_history(existing: list, new: list) -> list:
    """Reducer for the history field: LangGraph normally OVERWRITES a field
    each time a node returns it. This function tells LangGraph to instead
    APPEND new entries onto the existing list — which is what "memory
    across turns" actually requires. Without this, each turn's history
    would wipe out the previous turn's, defeating the whole point."""
    return existing + new


class AgentState(TypedDict):
    # The user's question for this turn, as typed.
    question: str

    # Set by the supervisor after classifying the question.
    route: Optional[Literal["sql", "rag", "both", "off_topic"]]

    # Populated by the SQL sub-agent, if it ran.
    sql_answer: Optional[str]
    sql_query: Optional[str]  # the actual SQL that was executed, for transparency

    # Populated by the RAG sub-agent, if it ran.
    rag_answer: Optional[str]
    rag_citations: Optional[list]

    # Set by the aggregator at the end — this is what the user actually sees.
    final_answer: Optional[str]

    # Set by the guardrail node — True if this turn was blocked before
    # ever reaching the supervisor/sub-agents.
    blocked: Optional[bool]

    # This conversation's (thread_id-scoped) question/answer history —
    # see _add_history above for why it appends instead of overwriting.
    history: Annotated[List[dict], _add_history]

    # Set by load_memory_node, once per turn, right after the guardrail --
    # semantically-relevant Q&A pairs pulled from THIS user's long-term
    # Store (across ALL past conversations, not just this thread_id). This
    # is what makes aggregator_node's store.put() writes actually useful
    # instead of write-only: the supervisor and both sub-agents read this
    # back when reasoning about the current question.
    long_term_memory: Optional[str]


def format_history(history: list, max_turns: int = 3) -> str:
    """Shared helper: turns the last few (question, answer) pairs into a
    plain-text block for prompts. Returns "(none)" for a fresh conversation.
    Used by the supervisor and both sub-agents so follow-up questions can
    be understood in context."""
    if not history:
        return "(none)"
    recent = history[-max_turns:]
    return "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in recent)
