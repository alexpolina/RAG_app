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
# Page config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat – Corvinus",
    page_icon="🤖",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────────────────────
# Custom CSS: dark theme, neon accents, futuristic font
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("""
  <style>
    /* Import futuristic font from Google */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');

    /* Base styles */
    body, .stApp {
      background-color: #0a0a0a;
      color: #e0e0e0;
      font-family: 'Roboto Mono', monospace;
    }
    /* Step containers */
    .step-container {
      background: linear-gradient(135deg, #111111 0%, #1e1e1e 100%);
      border: 1px solid #222222;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }
    /* Step headers */
    .step-header {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.6rem;
      color: #00f6ff;
      margin-bottom: 1rem;
    }
    /* File uploader box */
    .stFileUploader>div {
      border: 2px dashed #333333;
      border-radius: 8px;
    }
    /* Chat input style */
    .stChatInput>div>div>input {
      background-color: #1e1e1e !important;
      border: 1px solid #333333;
      border-radius: 8px;
      padding: 0.75rem;
      color: #e0e0e0;
    }
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
      border-radius: 12px !important;
      padding: 0.75rem !important;
      margin-bottom: 0.5rem !important;
    }
    /* User vs Assistant bubble colors */
    [data-testid="stChatMessage"] .message-content:has(+ .avatar) {
      background-color: #022b3a !important;  /* user bubble */
    }
    [data-testid="stChatMessage"] .message-content {
      background-color: #004e64 !important;  /* assistant bubble */
    }
  </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Silence internal warnings
# ────────────────────────────────────────────────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# ────────────────────────────────────────────────────────────────────────────────
# AIMLAPI client setup
# ────────────────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ────────────────────────────────────────────────────────────────────────────────
# Session state init & migration
# ────────────────────────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    # migrate any old tuple entries to dict
    migrated = []
    for e in st.session_state.chat_history:
        if isinstance(e, tuple) and len(e) == 2:
            migrated.append({"role": e[0], "content": e[1]})
        else:
            migrated.append(e)
    st.session_state.chat_history = migrated

# ────────────────────────────────────────────────────────────────────────────────
# Step 1: Upload & Index PDF (stacked)
# ────────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-container">', unsafe_allow_html=True)
st.markdown('<div class="step-header">📄 1. Upload & Index PDF</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Drag & drop a PDF here", type=["pdf"], help="Up to 200 MB")
if uploaded:
    st.info("Extracting text…")
    pdf_bytes = uploaded.read()
    text = load_pdf_text_from_memory(pdf_bytes)

    st.info("Splitting into chunks…")
    chunks = chunk_text(text)
    st.success(f"{len(chunks)} chunks created.")

    st.info("Generating embeddings…")
    vectors = get_embeddings(chunks, model="text-embedding-ada-002")
    if vectors:
        arr = np.array(vectors, dtype=np.float32)
        vs = VectorStore(arr.shape[1])
        vs.add_embeddings(arr, chunks)
        st.session_state.vector_store = vs
        st.success("🚀 Indexed and ready to chat!")
    else:
        st.error("❌ Embedding failed. Is the PDF text readable?")

st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Step 2: Chat with the PDF (stacked)
# ────────────────────────────────────────────────────────────────────────────────
if st.session_state.vector_store:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">💬 2. Chat with the PDF</div>', unsafe_allow_html=True)

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input (Enter to send)
    user_q = st.chat_input("Type your question…")
    if user_q:
        # Save & render user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        # Retrieve and generate
        q_vec = get_embeddings([user_q], model="text-embedding-ada-002")
        arr = np.array(q_vec, dtype=np.float32)
        results = st.session_state.vector_store.search(arr, k=5)
        context = "\n\n".join(chunk for chunk, _ in results)

        prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer:

{context}

QUESTION:
{user_q}
"""

        st.info("Generating answer…")
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that uses PDF context."},
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

        # Save & render assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    st.markdown('</div>', unsafe_allow_html=True)
