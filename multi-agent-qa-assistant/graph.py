from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from state import AgentState
from guardrails import guardrail_node
from memory import load_memory_node
from supervisor import supervisor_node
from agents.sql_node import sql_node
from agents.rag_node import rag_node
from aggregator import aggregator_node, off_topic_node


_memory_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def after_guardrail_decision(state: AgentState) -> str:
    return "end" if state.get("blocked") else "load_memory"


def route_decision(state: AgentState) -> str:
    return state["route"]


def after_sql_decision(state: AgentState) -> str:
    return "rag_node" if state["route"] == "both" else "aggregate"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_node", sql_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("off_topic", off_topic_node)
    graph.add_node("aggregate", aggregator_node)

    graph.add_edge(START, "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        after_guardrail_decision,
        {"end": END, "load_memory": "load_memory"},
    )

    # load_memory needs the Store, so it must run as a graph node (not
    # inline in the guardrail) -- LangGraph injects `store` automatically
    # into any node function that declares a `store: BaseStore` kwarg.
    graph.add_edge("load_memory", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "sql": "sql_node",
            "rag": "rag_node",
            "both": "sql_node",
            "off_topic": "off_topic",
        },
    )

    graph.add_conditional_edges(
        "sql_node",
        after_sql_decision,
        {"rag_node": "rag_node", "aggregate": "aggregate"},
    )

    graph.add_edge("rag_node", "aggregate")

    graph.add_edge("off_topic", END)
    graph.add_edge("aggregate", END)

    checkpointer = MemorySaver()
    store = InMemoryStore(
        index={
            "dims": 384,
            "embed": _memory_embeddings,
            "fields": ["question", "answer"],
        }
    )
    return graph.compile(checkpointer=checkpointer, store=store)
