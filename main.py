# main.py

import logging
import streamlit as st
import numpy as np

from openai import OpenAI, PermissionDeniedError
from transformers.pipelines import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# ────────────────────────────────────────────────────────────────────────────────
# Silence noisy warnings
# ────────────────────────────────────────────────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# ────────────────────────────────────────────────────────────────────────────────
# Configure AIMLAPI client once
# ────────────────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ────────────────────────────────────────────────────────────────────────────────
# Page header
# ────────────────────────────────────────────────────────────────────────────────
st.title("RAG Chat for Corvinus University")
st.write("Upload a PDF, then chat with it. Press Enter to send each question.")

# ────────────────────────────────────────────────────────────────────────────────
# Session state init & migration
# ────────────────────────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    # Migrate old tuple entries, if any
    migrated = []
    for e in st.session_state.chat_history:
        if isinstance(e, tuple) and len(e) == 2:
            migrated.append({"role": e[0], "content": e[1]})
        else:
            migrated.append(e)
    st.session_state.chat_history = migrated

# ────────────────────────────────────────────────────────────────────────────────
# Step 1: Upload & Index PDF
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("1️⃣ Upload & Index PDF", expanded=True):
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded:
        pdf_bytes = uploaded.read()
        st.info("Extracting text…")
        text = load_pdf_text_from_memory(pdf_bytes)

        st.info("Chunking…")
        chunks = chunk_text(text)
        st.info(f"{len(chunks)} chunks created.")

        st.info("Embedding…")
        vectors = get_embeddings(chunks, model="text-embedding-ada-002")
        if vectors:
            arr = np.array(vectors, dtype=np.float32)
            vs = VectorStore(arr.shape[1])
            vs.add_embeddings(arr, chunks)
            st.session_state.vector_store = vs
            st.success("Indexed – ready to chat!")
        else:
            st.error("No embeddings – is the PDF text readable?")

# ────────────────────────────────────────────────────────────────────────────────
# Step 2: Chat with chat_input + chat_message
# ────────────────────────────────────────────────────────────────────────────────
if st.session_state.vector_store:
    st.markdown("---")
    st.header("2️⃣ Chat with the PDF")

    # 1) Render past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 2) Get new question (Enter to send)
    user_q = st.chat_input("Your question:")
    if user_q:
        # a) Save user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        # b) Retrieve relevant chunks
        q_vec = get_embeddings([user_q], model="text-embedding-ada-002")
        q_arr = np.array(q_vec, dtype=np.float32)
        results = st.session_state.vector_store.search(q_arr, k=5)
        context = "\n\n".join(chunk for chunk, _ in results)

        # c) Build prompt
        prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer the user's question.

CONTEXT:
{context}

QUESTION:
{user_q}
"""

        st.info("Generating answer…")

        # d) Call AIMLAPI (with fallback)
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "You are an AI assistant backed by PDF context."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
            )
            answer = resp.choices[0].message.content

        except PermissionDeniedError:
            st.warning("Quota exceeded—falling back to local gpt2-xl.")
            gen = pipeline("text-generation", model="gpt2-xl", device=-1)
            fb = f"Context:\n{context}\n\nQ: {user_q}\nA:"
            out = gen(fb, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            answer = full[len(fb):].strip()

        # e) Save and render assistant message immediately below
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
