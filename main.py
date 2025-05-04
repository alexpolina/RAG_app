import logging
import streamlit as st
import numpy as np
from openai import OpenAI, PermissionDeniedError
from transformers.pipelines import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# Page setup
st.set_page_config(
    page_title="RAG Chat for Corvinus University",
    page_icon="🤖",
    layout="wide",
)
st.title("RAG Chat for Corvinus University")

# Custom styling: light theme, neon accents, futuristic font, narrower layout
st.markdown("""
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');
    .block-container {
      padding-top: 0rem !important;
      padding-bottom: 2rem !important;
      max-width: 800px !important;
      margin: auto !important;
    }
    body, .stApp {
      background-color: #f9f9f9;
      color: #333;
      font-family: 'Roboto Mono', monospace;
    }
    .step-container {
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .step-header {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.6rem;
      color: #0077cc;
      margin-bottom: 1rem;
    }
    .stFileUploader>div {
      border: 2px dashed #ccc;
      border-radius: 8px;
    }
    .stChatInput>div>div>input {
      background-color: #fff !important;
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 0.75rem;
      color: #333;
    }
    [data-testid="stChatMessage"] {
      border-radius: 12px !important;
      padding: 0.75rem !important;
      margin-bottom: 0.5rem !important;
      max-width: 80%;
    }
    [data-testid="stChatMessage"] .avatar + .message-content {
      background-color: #e6f7ff !important;
      border: 1px solid #99d6ff !important;
    }
    [data-testid="stChatMessage"] .message-content {
      background-color: #f0f0f0 !important;
      border: 1px solid #ddd !important;
    }
  </style>
""", unsafe_allow_html=True)

# Quiet warnings
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# API client
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# Session init and migration
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    migrated = []
    for e in st.session_state.chat_history:
        if isinstance(e, tuple) and len(e) == 2:
            migrated.append({"role": e[0], "content": e[1]})
        else:
            migrated.append(e)
    st.session_state.chat_history = migrated

# Upload & index
st.markdown('<div class="step-container">', unsafe_allow_html=True)
st.markdown('<div class="step-header">📄 1. Upload & Index PDF</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag & drop a PDF here", type=["pdf"], help="Up to 200 MB")
if uploaded:
    pdf_bytes = uploaded.read()
    text = load_pdf_text_from_memory(pdf_bytes)
    chunks = chunk_text(text)
    st.write(f"{len(chunks)} chunks created.")
    vectors = get_embeddings(chunks, model="text-embedding-ada-002")
    if vectors:
        arr = np.array(vectors, dtype=np.float32)
        vs = VectorStore(arr.shape[1])
        vs.add_embeddings(arr, chunks)
        st.session_state.vector_store = vs
        st.success("🚀 PDF indexed! Ready to chat.")
    else:
        st.error("Embedding failed. Is your PDF text readable?")
st.markdown('</div>', unsafe_allow_html=True)

# Chat interface
if st.session_state.vector_store:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">💬 2. Chat with the PDF</div>', unsafe_allow_html=True)

    # Display past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # New question input
    user_q = st.chat_input("Type your question…")
    if user_q:
        # Record user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        # Embed and retrieve
        q_vec = get_embeddings([user_q], model="text-embedding-ada-002")
        arr = np.array(q_vec, dtype=np.float32)
        results = st.session_state.vector_store.search(arr, k=5)
        context = "\n\n".join(chunk for chunk, _ in results)

        # Build and send prompt
        prompt = f"""
Use ONLY the following context to answer:

{context}

QUESTION:
{user_q}
"""
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "You are an AI assistant using PDF context."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
            )
            answer = resp.choices[0].message.content

        except PermissionDeniedError:
            st.warning("Quota reached—falling back to local gpt2-xl.")
            gen = pipeline("text-generation", model="gpt2-xl", device=-1)
            fb = f"Context:\n{context}\n\nQ: {user_q}\nA:"
            out = gen(fb, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            answer = full[len(fb):].strip()

        # Record assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    st.markdown('</div>', unsafe_allow_html=True)
