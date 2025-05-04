import logging
import streamlit as st
import numpy as np
from openai import OpenAI, PermissionDeniedError
from transformers.pipelines import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat for Corvinus University",
    page_icon="🤖",
    layout="wide",
)

# ── Title ───────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="title-container">
      <h1>RAG Chat for Corvinus University</h1>
      <p class="subtitle">Upload a PDF and ask any question about its contents</p>
    </div>
""", unsafe_allow_html=True)

# ── Styling ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');

  /* Center everything and cap width */
  .block-container {
    max-width: 900px !important;
    margin: 1rem auto !important;
    padding: 0 !important;
  }

  /* Title */
  .title-container h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5rem !important;
    text-align: center;
    margin-bottom: 0.2rem;
    color: #0077cc;
  }
  .title-container .subtitle {
    font-family: 'Roboto Mono', monospace;
    font-size: 1rem;
    text-align: center;
    color: #555;
    margin-bottom: 2rem;
  }

  /* Base text */
  body, .stApp {
    background-color: #f9f9f9;
    color: #333;
    font-family: 'Roboto Mono', monospace;
  }

  /* Section headers */
  h2 {
    font-family: 'Orbitron', sans-serif;
    color: #0077cc;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-size: 1.5rem !important;
  }

  /* File uploader */
  .stFileUploader>div {
    border: 2px dashed #ccc !important;
    border-radius: 6px !important;
    padding: 1rem !important;
  }

  /* Chat input */
  .stChatInput>div>div>input {
    max-width: 70%;
    margin: 1rem auto;
    display: block;
    background-color: #fff !important;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 0.6rem;
  }

  /* Chat bubbles */
  [data-testid="stChatMessage"] {
    border-radius: 10px !important;
    padding: 0.6rem !important;
    margin: 0.4rem 0 !important;
    max-width: 75%;
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

# ── Silence warnings ────────────────────────────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# ── API Client ─────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ── Session state ───────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    # migrate any old tuple entries
    mig = []
    for e in st.session_state.chat_history:
        if isinstance(e, tuple) and len(e) == 2:
            mig.append({"role": e[0], "content": e[1]})
        else:
            mig.append(e)
    st.session_state.chat_history = mig

# ── 1. Upload PDF ───────────────────────────────────────────────────────────────
st.header("1. Upload PDF")
uploaded = st.file_uploader("Drag & drop a PDF or click to browse", type=["pdf"])
if uploaded:
    data = uploaded.read()
    text = load_pdf_text_from_memory(data)
    chunks = chunk_text(text)
    st.success(f"{len(chunks)} chunks created.")
    embs = get_embeddings(chunks, model="text-embedding-ada-002")
    if embs:
        arr = np.array(embs, dtype=np.float32)
        vs = VectorStore(arr.shape[1])
        vs.add_embeddings(arr, chunks)
        st.session_state.vector_store = vs
        st.success("PDF indexed! You can now chat below.")
    else:
        st.error("Embedding failed—please check the PDF.")

# ── 2. Chat with PDF ────────────────────────────────────────────────────────────
if st.session_state.vector_store:
    st.header("2. Chat with the PDF")

    # show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ask
    q = st.chat_input("Type your question…")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        vec = get_embeddings([q], model="text-embedding-ada-002")
        arr = np.array(vec, dtype=np.float32)
        top5 = st.session_state.vector_store.search(arr, k=5)
        ctx = "\n\n".join(c for c, _ in top5)

        prompt = f"""Use ONLY the context below to answer:

{ctx}

QUESTION:
{q}
"""
        try:
            r = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "Answer using only the provided PDF context."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
            )
            ans = r.choices[0].message.content
        except PermissionDeniedError:
            gen = pipeline("text-generation", model="gpt2-xl", device=-1)
            fb = f"Context:\n{ctx}\n\nQ: {q}\nA:"
            out = gen(fb, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            ans = full[len(fb):].strip()

        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.write(ans)
