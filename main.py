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

# ── Silence noisy warnings ─────────────────────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
tf_logging.set_verbosity_error()

# ── Configure AIMLAPI client once ──────────────────────────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ── Page header ─────────────────────────────────────────────────────────────────
st.title("RAG System for Corvinus University")
st.write("""
Upload a PDF, then chat with the assistant about its contents.  
Ask as many follow-up questions as you like!
""")

# ── Session state initialization ────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "chat_history" not in st.session_state:
    # Each entry is a dict: {"role": "user"|"assistant", "content": str}
    st.session_state["chat_history"] = []

# ── Step 1: PDF Upload & Indexing ───────────────────────────────────────────────
with st.expander("1️⃣ Upload and index PDF", expanded=True):
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
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

        if len(vectors) > 0:
            dim = vectors_np.shape[1]
            vs = VectorStore(dim)
            vs.add_embeddings(vectors_np, chunks)
            st.session_state["vector_store"] = vs
            st.success("Embedding & indexing complete!")
        else:
            st.error("Failed to create embeddings—check PDF content.")

# ── Step 2: Chat Interface ──────────────────────────────────────────────────────
if st.session_state["vector_store"] is not None:
    st.markdown("---")
    st.header("2️⃣ Ask questions")

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    # Input for next question
    user_input = st.text_input("Your question:", key="new_question")
    if user_input:
        # Save user message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # 1) Embed and retrieve
        q_vec = get_embeddings([user_input], model="text-embedding-ada-002")
        q_np = np.array(q_vec, dtype=np.float32)
        results = st.session_state["vector_store"].search(q_np, k=5)
        context = "\n\n".join(chunk for chunk, _ in results)

        # 2) Build LLM prompt
        prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer the user's question.

CONTEXT:
{context}

QUESTION:
{user_input}
"""

        # 3) Call AIMLAPI (or fallback)
        try:
            resp = client.chat.completions.create(
                model="openai/o4-mini-2025-04-16",
                messages=[
                    {"role": "system",    "content": "You are an AI assistant that uses PDF context."},
                    {"role": "user",      "content": prompt},
                ],
                temperature=0,
            )
            answer = resp.choices[0].message.content

        except PermissionDeniedError:
            st.warning("Quota exceeded; falling back to local model (gpt2-xl).")
            gen = pipeline("text-generation", model="gpt2-xl", device=-1)
            fallback = f"Context:\n{context}\n\nQ: {user_input}\nA:"
            out = gen(fallback, max_new_tokens=50, truncation=True, do_sample=False)
            full = out[0]["generated_text"]
            answer = full[len(fallback):].strip()

        # 4) Save and display assistant message
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.markdown(f"**Assistant:** {answer}")

        # 5) Clear the input box
        st.session_state["new_question"] = ""
