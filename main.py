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

# ── Suppress noisy warnings ───────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# ── Configure AIMLAPI client ────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

st.title("RAG Chat for Corvinus University")
st.write("Upload a PDF and then ask as many follow-up questions as you like.")

# ── Initialize session state ────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    # list of (role, text)
    st.session_state.chat_history = []

# ── Step 1: Upload & Index ───────────────────────────────────────
with st.expander("1️⃣ Upload & Index PDF", expanded=True):
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None:
        pdf_bytes = uploaded.read()
        st.info("Extracting and chunking text…")
        text = load_pdf_text_from_memory(pdf_bytes)
        chunks = chunk_text(text)
        st.info(f"Split into {len(chunks)} chunks. Creating embeddings…")
        vectors = get_embeddings(chunks, model="text-embedding-ada-002")
        if vectors:
            arr = np.array(vectors, dtype=np.float32)
            vs = VectorStore(arr.shape[1])
            vs.add_embeddings(arr, chunks)
            st.session_state.vector_store = vs
            st.success("PDF indexed – ready for chat!")
        else:
            st.error("Failed to embed; is the PDF text readable?")

# Only show the chat form once the PDF is indexed
if st.session_state.vector_store:
    st.markdown("---")
    st.header("2️⃣ Chat with the PDF")

    # Display conversation so far
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    # Chat form
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input("Your question:", "")
        send = st.form_submit_button("Send")
    if send and user_q:
        # Save user message
        st.session_state.chat_history.append(("user", user_q))

        # Retrieve relevant chunks
        q_vec = get_embeddings([user_q], model="text-embedding-ada-002")
        q_arr = np.array(q_vec, dtype=np.float32)
        results = st.session_state.vector_store.search(q_arr, k=5)
        context = "\n\n".join(chunk for chunk, _ in results)

        # Build prompt
        prompt = (
            "You are a helpful AI assistant. Use ONLY the following context:\n\n"
            f"{context}\n\nQUESTION: {user_q}"
        )

        # Call AIMLAPI (with fallback)
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
            )
            answer = resp.choices[0].message.content

        except PermissionDeniedError:
            st.warning("Quota exceeded – using local gpt2-xl fallback.")
            gen = pipeline("text-generation", model="gpt2-xl", device=-1)
            fb_in = f"Context:\n{context}\n\nQ: {user_q}\nA:"
            out = gen(fb_in, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            answer = full[len(fb_in):].strip()

        # Save and display
        st.session_state.chat_history.append(("assistant", answer))
        st.markdown(f"**Assistant:** {answer}")
