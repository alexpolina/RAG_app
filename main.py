# main.py

import streamlit as st
import numpy as np

from openai import OpenAI, PermissionDeniedError
from transformers import pipeline

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# ─── Configure AIMLAPI client using Streamlit Secrets ────────────────────────
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

# ─── UI: Title & Instructions ────────────────────────────────────────────────
st.title("RAG System for Corvinus University")
st.write("""
1. Upload a PDF  
2. Ask a question about its content  
3. We'll help you find an answer based on the uploaded material.
""")

# Initialize vector store in session
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# ─── Step 1: Upload & Index PDF ───────────────────────────────────────────────
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
        dim = vectors_np.shape[1]
        vs = VectorStore(dim)
        vs.add_embeddings(vectors_np, chunks)
        st.session_state["vector_store"] = vs
        st.success("Embedding & indexing complete!")
    else:
        st.warning("No embeddings created — check PDF content.")

# ─── Step 2: Ask a Question & Retrieve Answer ────────────────────────────────
question = st.text_input("Ask a question about this PDF:")
if question and st.session_state["vector_store"]:

    # Embed the question
    question_vec = get_embeddings([question], model="text-embedding-ada-002")
    question_vec_np = np.array(question_vec, dtype=np.float32)

    # Retrieve top-k chunks
    results = st.session_state["vector_store"].search(question_vec_np, k=5)
    context_text = "\n\n".join([chunk for chunk, _ in results])

    # Build the prompt
    prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer the user's question.

CONTEXT:
{context_text}

QUESTION:
{question}
"""
    st.info("Generating final answer…")

    try:
        # Primary: AIMLAPI chat completions
        response = client.chat.completions.create(
            model="openai/o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": "You are an AI assistant that uses PDF context."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content

    except PermissionDeniedError:
        # Fallback: Local model via transformers
        st.warning("AIMLAPI quota exceeded; falling back to local model (distilgpt2).")
        gen = pipeline("text-generation", model="distilgpt2")
        # Short prompt for generation
        fallback_input = context_text + "\nQ: " + question + "\nA:"
        out = gen(fallback_input, max_length=len(fallback_input.split()) + 50, do_sample=False)
        # The pipeline returns the full text; strip prompt
        answer = out[0]["generated_text"][len(fallback_input):].strip()

    # Display answer
    st.markdown(f"**Answer:** {answer}")

    # Show which chunks were used
    with st.expander("Top Relevant Chunks"):
        for idx, (chunk, dist) in enumerate(results, start=1):
            st.write(f"**Rank {idx}** (distance {dist:.2f})")
            st.write(chunk[:300].replace("\n", " ") + "…")
