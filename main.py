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
# Configure AIMLAPI client
# ────────────────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ────────────────────────────────────────────────────────────────────────────────
# UI Header
# ────────────────────────────────────────────────────────────────────────────────
st.title("RAG Chat for Corvinus University")
st.write("""
1. Upload a PDF  
2. Ask as many follow-up questions as you like about its contents.
""")

# ────────────────────────────────────────────────────────────────────────────────
# Initialize / migrate session state
# ────────────────────────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Migrate old tuple-based chat entries to dict-based
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    migrated = []
    for entry in st.session_state.chat_history:
        # tuple format -> dict format
        if isinstance(entry, tuple) and len(entry) == 2:
            migrated.append({"role": entry[0], "content": entry[1]})
        # already dict -> keep
        elif isinstance(entry, dict):
            migrated.append(entry)
    st.session_state.chat_history = migrated

# ────────────────────────────────────────────────────────────────────────────────
# Step 1: Upload & Index PDF
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("1️⃣ Upload & Index PDF", expanded=True):
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None:
        pdf_bytes = uploaded.read()
        st.info("Extracting text from PDF…")
        text = load_pdf_text_from_memory(pdf_bytes)

        st.info("Splitting into chunks…")
        chunks = chunk_text(text)
        st.info(f"PDF split into {len(chunks)} chunks.")

        st.info("Generating embeddings…")
        vectors = get_embeddings(chunks, model="text-embedding-ada-002")
        if vectors:
            arr = np.array(vectors, dtype=np.float32)
            vs = VectorStore(arr.shape[1])
            vs.add_embeddings(arr, chunks)
            st.session_state.vector_store = vs
            st.success("PDF indexed—ready for chat!")
        else:
            st.error("Failed to embed PDF; please check its content.")

# ────────────────────────────────────────────────────────────────────────────────
# Step 2: Chat Interface
# ────────────────────────────────────────────────────────────────────────────────
if st.session_state.vector_store:
    st.markdown("---")
    st.header("2️⃣ Chat with the PDF")

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input("Your question:", "")
        submitted = st.form_submit_button("Send")

    if submitted and user_q:
        # Save user message
        st.session_state.chat_history.append({"role": "user",      "content": user_q})

        # Retrieve relevant chunks
        q_vec = get_embeddings([user_q], model="text-embedding-ada-002")
        q_arr = np.array(q_vec, dtype=np.float32)
        results = st.session_state.vector_store.search(q_arr, k=5)
        context = "\n\n".join(chunk for chunk, _ in results)

        # Build prompt for LLM
        prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer the user's question.

CONTEXT:
{context}

QUESTION:
{user_q}
"""

        # Try AIMLAPI chat completion
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system",    "content": "You are an AI assistant backed by PDF context."},
                    {"role": "user",      "content": prompt},
                ],
                temperature=0,
            )
            answer = resp.choices[0].message.content

        except PermissionDeniedError:
            # Fallback to local model
            st.warning("Quota exceeded—falling back to local gpt2-xl.")
            gen = pipeline("text-generation", model="gpt2-xl", device=-1)
            fb_input = f"Context:\n{context}\n\nQ: {user_q}\nA:"
            out = gen(fb_input, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            answer = full[len(fb_input):].strip()

        # Save and render assistant reply
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
