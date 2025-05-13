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

# Global styling
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');

  .block-container {
    max-width: 900px !important;
    margin: auto !important;
    padding: 2rem 2rem;
  }
  .stApp {
    background-color: #f9f9f9;
    font-family: 'Roboto Mono', monospace;
    color: #333;
  }

  /* Title */
  .title-container {
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .title-container h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.85rem;
    color: #0077cc;
    margin-bottom: 0.5rem;
    white-space: normal;
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
    margin-bottom: 1.5rem;
  }

  /* Tips Box */
  .tips-box {
    background-color: #e8f4ff;
    border-left: 4px solid #0077cc;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1.5rem;
  }

  h2 {
    font-family: 'Orbitron', sans-serif;
    color: #0077cc;
    margin-top: 2rem;
    font-size: 1.4rem;
  }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-container">
  <h1>RAG Chat for Corvinus University</h1>
  <div class="underline"></div>
  <p class="subtitle">Upload a PDF and ask any question about its contents</p>
</div>
""", unsafe_allow_html=True)

# Tips
st.markdown("""
<div class="tips-box">
  <strong>Tips:</strong>
  <ul>
    <li>Make sure your PDF contains selectable text (not scanned images).</li>
    <li>Wait for chunks to be created before asking questions.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# Hide noisy logs
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# API client
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
st.header("1. Upload your PDF")
uploaded = st.file_uploader("Upload course material as PDF", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    text = load_pdf_text_from_memory(pdf_bytes)
    chunks = chunk_text(text)
    st.success(f"✅ {len(chunks)} text chunks created")

    vectors = get_embeddings(chunks, model="text-embedding-ada-002")
    if vectors:
        arr = np.array(vectors, dtype=np.float32)
        vs = VectorStore(arr.shape[1])
        vs.add_embeddings(arr, chunks)
        st.session_state.vector_store = vs
        st.success("✅ PDF indexed! You can now chat below.")
    else:
        st.error("Failed to generate embeddings. Please check your PDF.")

# Chat interface
if st.session_state.vector_store:
    st.header("2. Ask questions about your PDF")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Type your question…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        q_vec = get_embeddings([question], model="text-embedding-ada-002")
        arr = np.array(q_vec, dtype=np.float32)
        top5 = st.session_state.vector_store.search(arr, k=5)
        context = "\n\n".join(c for c, _ in top5)

        prompt = f"""Use ONLY the context below to answer:

{context}

QUESTION:
{question}
"""
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "Answer using only the provided PDF context."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
            )
            answer = resp.choices[0].message.content
        except PermissionDeniedError:
            gen = pipeline("text-generation", model="gpt2-xl", device=-1)
            fb = f"Context:\n{context}\n\nQ: {question}\nA:"
            out = gen(fb, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            answer = full[len(fb):].strip()

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
