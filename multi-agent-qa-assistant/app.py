"""
Multi-Source Agentic Q&A Assistant — chat interface.

A thin client: this file does NOT build or run the LangGraph graph
itself. All of that (guardrail -> supervisor -> sql/rag/both/off_topic ->
aggregate, the long-term memory store, everything) lives in
backend_api.py, dockerized as its own service (see Dockerfile.backend /
docker-compose.yml). This file just sends the user's question to that
backend over HTTP and displays whatever comes back -- so this is the one
piece of the whole system that stays local / undockerized.

Run with: streamlit run app.py
Point BACKEND_URL at wherever the backend is reachable -- defaults to
http://localhost:8001, which matches docker-compose.yml's port mapping,
so no extra config is needed when running everything on one machine.
"""

import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

st.set_page_config(page_title="Multi-Source Q&A Assistant", page_icon="🔎")
st.title("Multi-Source Agentic Q&A Assistant")
st.caption("Ask about the Chinook music-store data or the biomedical research literature.")

if "history" not in st.session_state:
    st.session_state.history = []

# thread_id = THIS conversation only (short-term memory, scoped per browser tab).
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


if "user_id" not in st.session_state:
    st.session_state.user_id = "anonymous"


ROUTE_LABELS = {
    "sql": "SQL agent",
    "rag": "RAG agent",
    "both": "SQL + RAG agents",
    "off_topic": "Guardrail (off-topic)",
}


def render_turn(question: str, route: str, answer: str):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Handled by: **{ROUTE_LABELS.get(route, route)}**")


with st.sidebar:
    st.subheader("Identity")
    user_id_input = st.text_input(
        "Your User ID",
        value=st.session_state.user_id,
        help="Type any name/ID. Reusing the same one later reconnects you "
             "to your own long-term Q&A history, even in a new conversation.",
    )
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input or "anonymous"

    st.divider()
    st.caption(f"Conversation: `{st.session_state.thread_id[:8]}`")
    if st.button("Start new conversation"):
        # Resets the CONVERSATION only — user_id (long-term identity) is
        # deliberately left untouched, so the backend's memory of this
        # user survives across this reset.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("Your long-term history")
    st.caption("Everything you've ever asked, across ALL conversations — "
               "not just this one.")
    try:
        resp = requests.get(f"{BACKEND_URL}/history/{st.session_state.user_id}", timeout=10)
        resp.raise_for_status()
        past_items = resp.json()
    except requests.RequestException:
        past_items = None

    if past_items is None:
        st.caption("Couldn't reach the backend to load your history.")
    elif not past_items:
        st.caption("Nothing recorded yet for this User ID.")
    else:
        for item in past_items:
            st.caption(f"[{item['route']}] {item['question']}")

# Replay prior turns in this session
for turn in st.session_state.history:
    render_turn(turn["question"], turn["route"], turn["answer"])

question = st.chat_input("Ask about artists, tracks, invoices, or biomedical research...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Routing and retrieving..."):
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "question": question,
                        "thread_id": st.session_state.thread_id,
                        "user_id": st.session_state.user_id,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
            route = result["route"]
            answer = result["final_answer"]
            st.write(answer)
            st.caption(f"Handled by: **{ROUTE_LABELS.get(route, route)}**")
        except requests.RequestException:
            # Catches anything from a connection refused (backend not
            # running) to a 5xx from the graph itself, so the UI degrades
            # gracefully instead of showing a raw traceback.
            route = "error"
            answer = (
                "Something went wrong processing that question. "
                "Please try again — if this keeps happening, the "
                "backend service may be temporarily unavailable."
            )
            st.error(answer)

    st.session_state.history.append({"question": question, "route": route, "answer": answer})
