# main.py

import streamlit as st
import numpy as np
from openai import OpenAI

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# Configure AIMLAPI client once
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["TEXT_API_KEY"],
)

st.title("RAG System for Corvinus University")
st.write("""
1. Upload a PDF  
2. Ask a question about its content  
3. We'll help you find an answer based on the uploaded material.
""")

# Initialize vector store in session
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# Step 1: Upload and index PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    st.info("Extracting text from PDF…")
    pdf_text = load_pdf_text_from_memory(pdf_bytes)

    st.info("Splitting into chunks…")
    chunks = chunk_text(pdf_text)
    st.info(f"PDF split into {len(chunks)} chunks")

    st.info("Generating embeddings…")
    # Use the valid embedding model name here
    vectors = get_embeddings(chunks, model="text-embedding-ada-002")
    vectors_np = np.array(vectors, dtype=np.float32)

    if len(vectors) > 0:
        dim = vectors_np.shape[1]
        vs = VectorStore(dim)
        vs.add_embeddings(vectors_np, chunks)
        st.session_state["vector_store"] = vs
        st.success("Embedding & indexing complete!")
    else:
        st.warning("No embeddings created — check PDF content.")

# Step 2: Ask question & generate answer
question = st.text_input("Ask a question about this PDF:")
if question and st.session_state["vector_store"] is not None:
    # Embed the question using the same model
    question_vec = get_embeddings([question], model="text-embedding-ada-002")
    question_vec_np = np.array(question_vec, dtype=np.float32)

    # Retrieve top-k chunks
    results = st.session_state["vector_store"].search(question_vec_np, k=5)
    context_text = "\n\n".join([chunk for chunk, _ in results])

    # Build prompt
    prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer the user's question.

CONTEXT:
{context_text}

QUESTION:
{question}
"""

    st.info("Generating final answer…")
    with st.spinner("Thinking…"):
        response = client.chat.completions.create(
            model="openai/o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": "You are an AI assistant that uses PDF context."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
        )

    answer = response.choices[0].message.content
    st.markdown(f"**Answer:** {answer}")

    # Show which chunks were used
    with st.expander("Top Relevant Chunks"):
        for idx, (chunk, dist) in enumerate(results, start=1):
            st.write(f"**Rank {idx}** (distance {dist:.2f})")
            st.write(chunk[:300].replace("\n", " ") + "…")
