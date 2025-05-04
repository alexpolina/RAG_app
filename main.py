# main.py

import logging
import streamlit as st
import numpy as np

from openai import OpenAI, PermissionDeniedError

# <-- Changed import from transformers.pipelines -->
from transformers.pipelines import pipeline
from transformers import logging as tf_logging

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# ── Silence warnings ─────────────────────────────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# ── Configure AIMLAPI client ─────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ── UI: Title ─────────────────────────────────────────────────────────────────────
st.title("RAG System for Corvinus University")
st.write("""
1. Upload a PDF  
2. Ask a question about its content  
3. We'll help you find an answer based on the uploaded material.
""")

# Store vector index in session
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# ── Step 1: Upload & index PDF ────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    st.info("Extracting text from PDF…")
    pdf_text = load_pdf_text_from_memory(pdf_bytes)

    st.info("Splitting into chunks…")
    chunks = chunk_text(pdf_text)
    st.info(f"PDF split into {len(chunks)} chunks")

    st.info("Generating embeddings…")
    vectors = get_embeddings(chunks, model="text-embedding-ada-002")
    vectors_np = np.array(vectors, dtype=np.float32)

    if vectors:
        vs = VectorStore(vectors_np.shape[1])
        vs.add_embeddings(vectors_np, chunks)
        st.session_state["vector_store"] = vs
        st.success("Embedding & indexing complete!")
    else:
        st.warning("No embeddings created — please check the PDF content.")

# ── Step 2: Ask & answer ─────────────────────────────────────────────────────────
question = st.text_input("Ask a question about this PDF:")
if question and st.session_state["vector_store"]:
    q_vec = get_embeddings([question], model="text-embedding-ada-002")
    q_np  = np.array(q_vec, dtype=np.float32)

    # retrieve
    results = st.session_state["vector_store"].search(q_np, k=5)
    context = "\n\n".join(chunk for chunk, _ in results)

    prompt = f"""
You are a helpful AI assistant. Use ONLY the following context:

{context}

Question: {question}
"""

    st.info("Generating answer…")
    try:
        resp = client.chat.completions.create(
            model="openai/o4-mini-2025-04-16",
            messages=[
                {"role":"system", "content":"Use only the context."},
                {"role":"user",   "content":prompt},
            ],
            temperature=0,
        )
        answer = resp.choices[0].message.content

    except PermissionDeniedError:
        st.warning("Quota hit – using local model fallback (gpt2-xl).")
        gen = pipeline("text-generation", model="gpt2-xl", device=-1)
        fb_input = f"Context:\n{context}\n\nQ: {question}\nA:"
        out = gen(fb_input, max_new_tokens=50, truncation=True, do_sample=False)
        full = out[0]["generated_text"]
        answer = full[len(fb_input):].strip()

    st.markdown(f"**Answer:** {answer}")

    with st.expander("Relevant Chunks"):
        for i, (chunk, dist) in enumerate(results, 1):
            st.write(f"**{i}.** (dist {dist:.2f}) {chunk[:200]}…")
