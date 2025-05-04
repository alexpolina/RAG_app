import logging
import streamlit as st
import numpy as np
from openai import OpenAI, PermissionDeniedError
from transformers.pipelines import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# — Page setup —
st.set_page_config(
    page_title="RAG Chat for Corvinus University",
    page_icon="🤖",
    layout="wide",
)
st.title("RAG Chat for Corvinus University")

# — Styling: lighter theme, neon headers, tighter width —
st.markdown("""
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');
    .block-container {
      padding-top: 0 !important;
      padding-bottom: 1rem !important;
      max-width: 640px !important;
      margin: 0 auto !important;
    }
    body, .stApp {
      background-color: #f9f9f9;
      color: #333;
      font-family: 'Roboto Mono', monospace;
    }
    .step-container {
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 1.2rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .step-header {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.5rem;
      color: #0077cc;
      margin-bottom: 0.8rem;
    }
    .stFileUploader>div {
      border: 2px dashed #ccc;
      border-radius: 6px;
    }
    .stChatInput>div>div>input {
      background-color: #fff !important;
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 0.6rem;
      color: #333;
    }
    [data-testid="stChatMessage"] {
      border-radius: 10px !important;
      padding: 0.6rem !important;
      margin-bottom: 0.4rem !important;
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

# quiet internal noise
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# prepare API client
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    # convert any old tuple entries
    migrated = []
    for entry in st.session_state.chat_history:
        if isinstance(entry, tuple):
            migrated.append({"role": entry[0], "content": entry[1]})
        else:
            migrated.append(entry)
    st.session_state.chat_history = migrated

# — 1️⃣ Upload & index PDF —
st.markdown('<div class="step-container">', unsafe_allow_html=True)
st.markdown('<div class="step-header">📄 1. Upload & Index PDF</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag & drop a PDF or click to browse", type=["pdf"])
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
        st.success("Indexed and ready to chat!")
    else:
        st.error("Embedding failed—check PDF content.")
st.markdown('</div>', unsafe_allow_html=True)

# — 2️⃣ Chat with the PDF —
if st.session_state.vector_store:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">💬 2. Chat with the PDF</div>', unsafe_allow_html=True)

    # show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # input and respond
    question = st.chat_input("Type your question here…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        q_vec = get_embeddings([question], model="text-embedding-ada-002")
        q_arr = np.array(q_vec, dtype=np.float32)
        results = st.session_state.vector_store.search(q_arr, k=5)
        context = "\n\n".join(chunk for chunk, _ in results)

        prompt = f"""You are a helpful AI assistant. Use ONLY the context below to answer:

{context}

QUESTION:
{question}
"""
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": "Use the provided PDF context only."},
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
            st.write(answer)

    st.markdown('</div>', unsafe_allow_html=True)
