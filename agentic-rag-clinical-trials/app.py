"""
day6: streamlit chat UI - the "frontend" container in the docker plan.
talks to the FastAPI backend (src/api.py) over plain HTTP, same as any
other client would - doesn't import the agent or touch a single tool
itself, so it stays a genuinely separate service.

run with: streamlit run app.py
(needs the backend running first - see src/api.py. BACKEND_URL points at
it: defaults to localhost for running everything on one machine,
docker-compose overrides it to the backend service's container name)
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="agenticrag", page_icon=":dna:")
st.title("agenticrag")
st.caption("Ask about clinical trials (AACT) or the ingested research papers.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None  # backend hands us one on the first reply

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question, "thread_id": st.session_state.thread_id},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                st.session_state.thread_id = data["thread_id"]
            except requests.RequestException as e:
                # resp.raise_for_status()'s own message is useless ("500
                # Server Error: Internal Server Error for url: ...") - the
                # actual reason (rate limited, tool error, whatever) is in
                # the backend's JSON body, so try to pull that out instead.
                # e.response is None only for an actual connection failure
                # (backend not running, wrong port) - if we got a response
                # at all the backend WAS reached, it just failed to answer
                # (usually a groq rate limit), so say that instead of
                # "couldn't reach" which was misleading in that case.
                if e.response is not None:
                    try:
                        detail = e.response.json().get("detail")
                    except ValueError:
                        detail = None
                    answer = f"backend returned an error: {detail or e}"
                else:
                    answer = f"couldn't reach the backend at {BACKEND_URL}: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.caption(f"backend: {BACKEND_URL}")
    # visible on purpose - this is what ties a follow-up question back to
    # the same working-memory state on the backend (see agent.py's
    # checkpointer). "new conversation" below clears it, which is also the
    # easiest way to show long-term memory recall working live: ask
    # something, hit new conversation, ask something related and it still
    # remembers even though this id changed.
    st.caption("thread id")
    st.code(st.session_state.thread_id or "(none yet - ask something first)", language=None)
    if st.button("new conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()
