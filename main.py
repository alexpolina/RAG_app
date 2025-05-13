import logging
import streamlit as st
import numpy as np
from openai import OpenAI, PermissionDeniedError
from transformers import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# Page configuration
st.set_page_config(
    page_title="RAG Chat for Corvinus University",
    page_icon="🤖",
    layout="wide",
)

# Global CSS styles
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Roboto+Mono&display=swap');

  .block-container {
    max-width: 960px !important;
    margin: 2rem auto;
    padding: 1rem;
  }

  body, .stApp {
    background-color: #f8f9fa;
    font-family: 'Roboto Mono', monospace;
  }

  .title-container {
    text-align: center;
    margin-bottom: 1rem;
  }

  .title-container h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.7rem;
    color: #0056b3;
    margin-bottom: 0.3rem;
  }

  .title-container .underline {
    height: 4px;
    width: 50%;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    margin: 0 auto 1rem auto;
    border-radius: 2px;
  }

  .subtitle {
    font-size: 1rem;
    color: #555;
  }

  .tips-box {
    background-color: #e7f3ff;
    border-left: 4px solid #0072ff;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1.5rem;
  }

  .tips-box li {
    margin-bottom: 0.5rem;
  }

  .stChatInput input {
    max-width: 80%;
    margin: 0 auto;
    border-radius: 6px;
  }

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-container">
  <h1>RAG Chat for Corvinus University</h1>
  <div class="underline"></div>
  <p class="subtitle">Upload a PDF and ask any question based on it.</p>
</div>
""", unsafe_allow_html=True)

# Optional help
with st.expander("❔ How to use this app", expanded=False):
    st.write("""
    1. Upload a course-related PDF file.
    2. The document will be split and indexed.
    3. Ask a question, and get an answer based only on your uploaded material.
    """)

st.markdown("""
<div class="tips-box">
<strong>Tips:</strong>
<ul>
  <li>Use a real text-based PDF (not scanned images).</li>
  <li>Wait for confirmation before asking a question.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# OpenAI client
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
st.header("1. Upload PDF")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
if uploaded:
    text = load_pdf_text_from_memory(uploaded.read())
    chunks = chunk_text(text)
    st.success(f"{len(chunks)} chunks created.")
    embeddings = get_embeddings(chunks, model="text-embedding-ada-002")
    if embeddings:
        matrix = np.array(embeddings, dtype=np.float32)
        vs = VectorStore(matrix.shape[1])
        vs.add_embeddings(matrix, chunks)
        st.session_state.vector_store = vs
        st.success("PDF indexed. You can now ask questions.")
    else:
        st.error("Could not create embeddings. Check PDF content.")

# Ask questions
if st.session_state.vector_store:
    st.header("2. Chat with the PDF")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        q_vec = get_embeddings([question], model="text-embedding-ada-002")
        top_k = st.session_state.vector_store.search(np.array(q_vec, dtype=np.float32), k=5)
        context = "\n\n".join([c for c, _ in top_k])

        prompt = f"""Use ONLY the context below to answer the question:

{context}

QUESTION:
{question}
"""

        try:
            response = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "Answer using only the provided PDF context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        except PermissionDeniedError:
            gen = pipeline("text-generation", model="distilgpt2", device=-1)
            fallback_input = f"{context}\nQ: {question}\nA:"
            out = gen(fallback_input, max_new_tokens=100, do_sample=False, truncation=True)
            full = out[0]["generated_text"]
            answer = full[len(fallback_input):].strip()

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
