import logging
import streamlit as st
import numpy as np
from openai import OpenAI, PermissionDeniedError
from transformers.pipelines import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat for Corvinus University",
    page_icon="🤖",
    layout="wide",
)

# ─── How-to expander ───────────────────────────────────────────────────────────
with st.expander("❔ How to use this app"):
    st.write("""
    1. Upload a PDF containing your course materials.  
    2. The app will split it into chunks and index them.  
    3. Ask any question in the chat box below.  
    4. Receive answers sourced only from your PDF.
    """)

# ─── Stylish header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-container">
  <h1>RAG Chat for Corvinus University</h1>
  <div class="underline"></div>
  <p class="subtitle">Upload a PDF and ask any question about its contents</p>
</div>
""", unsafe_allow_html=True)

# ─── Tips box above upload step ────────────────────────────────────────────────
st.markdown("""
<div class="tips-box">
  <strong>🔖 Tips:</strong>
  <ul>
    <li>Make sure your PDF text is selectable (not just scanned images).</li>
    <li>Max file size: 200 MB.</li>
    <li>Wait for “chunks created” before asking questions.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');

  /* Centered content with limits */
  .block-container {
    max-width: 900px !important;
    margin: 2rem auto 2rem auto !important;
    padding: 0 !important;
  }

  body, .stApp {
    background-color: #f9f9f9;
    color: #333;
    font-family: 'Roboto Mono', monospace;
  }

  /* Title styling */
  .title-container {
    text-align: center;
    margin-bottom: 1rem;
  }
  .title-container h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.25rem;
    color: #0077cc;
    margin: 0;
  }
  .title-container .underline {
    height: 4px;
    width: 60%;
    background: linear-gradient(to right, #00d2ff, #3a7bd5);
    margin: 0.4rem auto 1rem auto;
    border-radius: 2px;
  }
  .title-container .subtitle {
    font-size: 1rem;
    color: #555;
    margin: 0;
  }

  /* Tips box */
  .tips-box {
    background-color: #e8f4ff;
    border-left: 4px solid #0077cc;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    margin-left: 1rem;
  }
  .tips-box ul {
    margin: 0.5rem 0 0.5rem 1.2rem;
    padding: 0;
  }
  .tips-box li {
    margin-bottom: 0.4rem;
  }

  /* Section headers */
  h2 {
    font-family: 'Orbitron', sans-serif;
    color: #0077cc;
    margin-top: 2rem !important;
    margin-bottom: 0.75rem !important;
    font-size: 1.4rem !important;
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

# ─── Quiet warnings ─────────────────────────────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# ─── API client ─────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ─── Session state ──────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    # migrate old tuple entries if any
    migrated = []
    for e in st.session_state.chat_history:
        if isinstance(e, tuple) and len(e) == 2:
            migrated.append({"role": e[0], "content": e[1]})
        else:
            migrated.append(e)
    st.session_state.chat_history = migrated

# ─── 1. Upload PDF ───────────────────────────────────────────────────────────────
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
        st.success("✅ PDF indexed! You can now chat below.")
    else:
        st.error("⚠ Embedding failed—please check the PDF content.")

# ─── 2. Chat with the PDF ────────────────────────────────────────────────────────
if st.session_state.vector_store:
    st.header("2. Chat with the PDF")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    q = st.chat_input("Type your question…")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        vec = get_embeddings([q], model="text-embedding-ada-002")
        arr = np.array(vec, dtype=np.float32)
        top5 = st.session_state.vector_store.search(arr, k=5)
        ctx = "\n\n".join(c for c,_ in top5)

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
