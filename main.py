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
# Page configuration
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat – Corvinus",
    page_icon="📄",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────────────────────
# Custom CSS for modern styling
# ────────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Remove Streamlit header/footer */
    #MainMenu, footer, header { visibility: hidden; }
    /* Expand chat and upload containers */
    .stApp > div { padding: 1rem 2rem; }
    /* Style step headers */
    .step-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    /* Input and chat box styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 0.75rem;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    /* Chat bubble overrides */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        padding: 0.75rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Session state init
# ────────────────────────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ────────────────────────────────────────────────────────────────────────────────
# Layout: Two columns for Upload/Index vs Chat
# ────────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="step-header">📄 1. Upload & Index PDF</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag or browse a PDF", type=["pdf"], help="Max 200 MB")
    if uploaded:
        pdf_bytes = uploaded.read()
        st.info("Extracting text…")
        text = load_pdf_text_from_memory(pdf_bytes)
        st.info("Chunking text…")
        chunks = chunk_text(text)
        st.info(f"{len(chunks)} chunks created.")
        st.info("Generating embeddings…")
        vectors = get_embeddings(chunks, model="text-embedding-ada-002")
        if vectors:
            arr = np.array(vectors, dtype=np.float32)
            vs = VectorStore(arr.shape[1])
            vs.add_embeddings(arr, chunks)
            st.session_state.vector_store = vs
            st.success("✅ Indexed and ready to chat!")
        else:
            st.error("❌ Embedding failed. Check PDF content.")

with col2:
    st.markdown('<div class="step-header">💬 2. Chat with the PDF</div>', unsafe_allow_html=True)
    if not st.session_state.vector_store:
        st.info("Index a PDF first to start chatting.")
    else:
        # Render conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Input with Enter-to-send
        user_q = st.chat_input("Type your question…")
        if user_q:
            # Save user message
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.write(user_q)

            # Retrieve top-5 chunks
            q_vec = get_embeddings([user_q], model="text-embedding-ada-002")
            q_arr = np.array(q_vec, dtype=np.float32)
            results = st.session_state.vector_store.search(q_arr, k=5)
            context = "\n\n".join(chunk for chunk, _ in results)

            # Build prompt
            prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer:

{context}

QUESTION:
{user_q}
"""

            # Generate answer (with quota fallback)
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

            # Save & render assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
