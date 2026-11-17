from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from state import AgentState

MAX_MEMORIES = 3


def load_memory_node(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    question = state["question"]

    try:
        results = store.search((user_id, "qa_history"), query=question, limit=MAX_MEMORIES)
    except Exception:
        # Same defensive stance as everywhere else in this graph: a memory
        # lookup failing should degrade to "no memory available", never
        # crash the whole turn.
        results = []

    if not results:
        return {"long_term_memory": "(none)"}

    memory_text = "\n".join(
        f"Q: {r.value['question']}\nA: {r.value['answer']}" for r in results
    )
    return {"long_term_memory": memory_text}
