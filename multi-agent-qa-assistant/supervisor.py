"""
The supervisor node: a single routing decision made once per turn, right
after long-term memory is loaded. Classifies the question into "sql",
"rag", "both", or "off_topic" -- everything downstream (which sub-agent(s)
run, what the aggregator stitches together) branches off this one field.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from state import AgentState, format_history

# env var is called ANTHROPIC_API_KEY but it's actually the groq key (old name, kept it)
_api_key = os.getenv("ANTHROPIC_API_KEY")
_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=_api_key)


ROUTING_PROMPT = """You are a router for a Q&A system with two data sources:

1. SQL: a Chinook music-store relational database — artists, albums, tracks,
   invoices, customers, employees, playlists. Sales and catalog questions.
2. RAG: a biomedical research corpus — PubMed clinical trial documents
   covering topics like lupus, pediatrics, pneumonia, and pain research.

Recent conversation (may be "(none)" if this is the first question):
{history}

Relevant questions this user has asked in PAST conversations (may be
"(none)"):
{long_term_memory}

If the current question is a natural follow-up to that conversation — asking
for more detail, clarification, or continuing the same topic — classify it
based on the topic actually being followed up on, not the follow-up phrasing
alone. For example, if the prior turn discussed a lupus trial, "what page was
that on?" is "rag", not "off_topic", because it continues that same topic.

Classify the user's question into exactly one category:
- "sql"       -> answerable from the music-store database
- "rag"       -> answerable from the biomedical research documents
- "both"      -> genuinely needs information from both sources
- "off_topic" -> unrelated to either data source, even considering the
                 conversation above

Respond with only the single category word, nothing else.

Question: {question}
"""


def supervisor_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """Classifies state["question"] into a route and returns {"route": ...}.
    Falls back to "off_topic" if the LLM ever returns something outside the
    four expected categories, so a malformed response can never crash the
    graph or silently route somewhere unhandled."""
    question = state["question"]
    history_text = format_history(state.get("history", []))
    long_term_memory = state.get("long_term_memory") or "(none)"

    response = _llm.invoke(
        ROUTING_PROMPT.format(question=question, history=history_text, long_term_memory=long_term_memory),
        config=config,
    )
    route = response.content.strip().lower()

    if route not in {"sql", "rag", "both", "off_topic"}:
        route = "off_topic"

    return {"route": route}
