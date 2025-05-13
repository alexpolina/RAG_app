import logging
import streamlit as st
import numpy as np
from openai import OpenAI, PermissionDeniedError
from transformers import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# Page settings
st.set_page_config(
    page_title="RAG Chat for Corvinus University",
    page_icon="🤖",
    layout="wide",
)

# Custom styles
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');

  .block-container {
    max-width: 1100px !important;
    padding: 0 2rem !important;
  }
  body, .stApp {
    background-color: #f9f9f9;
    font-family: 'Roboto Mono', monospace;
    color: #333;
  }
  .title-container {
    text-align: center;
    padding-top: 1.5rem;
  }
  .title-container h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    color: #0077cc;
    margin-bottom: 0.3rem;
  }
  .underline {
    width: 60%;
    height: 4px;
    margin: 0.3rem auto 1rem auto;
    background: linear-gradient(to right, #00d2ff, #3a7bd5);
    border-radius: 2px;
  }
  .subtitle {
    color: #666;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }
  .tips-box {
    background: #e8f4ff;
    padding: 1rem;
    border-left: 4px solid #0077cc;
    border-radius: 6px;
    margin-bottom: 2rem;
  }
  .tips-box ul {
    margin-left: 1.2rem;
    padding-left: 0;
  }
  .tips-box li {
    margin-bottom: 0.5rem;
  }
  h2 {
    font-family: 'Orbitron', sans-serif;
    color: #0077cc;
    margin-top: 2.5rem;
    font-size: 1.4rem;
  }
  .stFileUploader>div {
    border: 2px dashed #ccc !important;
    border-radius: 6px !important;
    padding: 1rem !important;
  }
  .stChatInput>div>div>input {
    background-color: #fff !important;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 0.6rem;
    max-width: 700px;
    margin: 1rem auto;
    display: block;
  }
  [data-testid="stChatMessage"] {
    max-width: 75%;
    padding: 0.6rem;
    border-radius: 10px;
    margin-bottom: 0.4rem;
  }
</style>
""", unsafe_allow_html=True)

# Title block
st.markdown("""
<div class="title-container">
  <h1>RAG Chat for Corvinus University</h1>
  <div class="underline"></div>
  <div class="subtitle">Upload a PDF and ask any question about its contents</div>
</div>
""", unsafe_allow_html=True)

# How-to section
with st.expander("❔ How to use this app", expanded=False):
    st.write("""
    1. Upload a course-related PDF.  
    2. The system processes it into chunks and generates vector embeddings.  
    3. Ask a question below.  
    4. You’ll get answers based only on the uploaded content.
    """)

# Tips
st.markdown("""
<div class="tips-box">
  <strong>🔖 Tips:</strong>
  <ul>
    <li>Make sure your PDF contains selectable text (not scanned images).</li>
    <li>Wait for "PDF indexed" before asking anything.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# Logging cleanup
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# OpenAI client
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"]
)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload
st.header("1. Upload PDF")
uploaded = st.file_uploader("Drag & drop a PDF here", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    text = load_pdf_text_from_memory(pdf_bytes)
    chunks = chunk_text(text)
    st.success(f"{len(chunks)} chunks created.")
    vectors = get_embeddings(chunks, model="text-embedding-ada-002")
    if vectors:
        arr = np.array(vectors, dtype=np.float32)
        vs = VectorStore(arr.shape[1])
        vs.add_embeddings(arr, chunks)
        st.session_state.vector_store = vs
        st.success("✅ PDF indexed! Now you can chat.")
    else:
        st.error("❌ Could not generate embeddings.")

# Chat
if st.session_state.vector_store:
    st.header("2. Chat with the PDF")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    q = st.chat_input("Type your question here...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        q_vec = get_embeddings([q], model="text-embedding-ada-002")
        arr = np.array(q_vec, dtype=np.float32)
        top_chunks = st.session_state.vector_store.search(arr, k=5)
        context = "\n\n".join([chunk for chunk, _ in top_chunks])

        prompt = f"""Use ONLY the context below to answer:

{context}

Question:
{q}
"""
        try:
            response = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "Only answer based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            answer = response.choices[0].message.content
        except PermissionDeniedError:
            fb_prompt = f"{context}\n\nQ: {q}\nA:"
            fallback = pipeline("text-generation", model="gpt2", device=-1)
            out = fallback(fb_prompt, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            answer = full[len(fb_prompt):].strip()

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
