"""Streamlit chat UI for the biomedical RAG agent.

Run with: streamlit run streamlit_app.py
"""
import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.agent import ask

load_dotenv()

st.set_page_config(page_title="Biomedical Research Assistant", page_icon="🧬")
st.title("🧬 Biomedical Research Assistant")
st.caption("Answers are grounded only in the ingested PubMed PDF corpus.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    # Scopes conversation memory to this browser session — a page refresh
    # (which resets st.session_state) starts a fresh thread, same as the
    # visible chat history resetting.
    st.session_state.thread_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the ingested biomedical literature...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and verifying..."):
            try:
                answer = asyncio.run(ask(question, thread_id=st.session_state.thread_id))
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
