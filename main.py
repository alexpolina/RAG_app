# main.py
import streamlit as st
import numpy as np
import openai

from pdf_loader import load_pdf_text_from_memory, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

openai.api_base = "https://api.aimlapi.com/v1"
# We will set openai.api_key from st.secrets when needed

st.title("RAG System for Corvinus university")
st.write("""
1. Upload a PDF
2. Ask a question about its content
3. We'll retrieve relevant chunks and use an AIMLAPI model to generate a final answer.
""")

# Session-level vector store
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    st.info("Extracting text from PDF...")
    pdf_text = load_pdf_text_from_memory(pdf_bytes)  # ✅ fixed function

    chunks = chunk_text(pdf_text)
    st.info(f"PDF split into {len(chunks)} chunks...")

    st.info("Embedding chunks...")
    vectors = get_embeddings(chunks, model="embedding-4o-latest")
    vectors_np = np.array(vectors, dtype=np.float32)

    if len(vectors) > 0:
        dim = vectors_np.shape[1]
        vs = VectorStore(dim)
        vs.add_embeddings(vectors_np, chunks)
        st.session_state["vector_store"] = vs
        st.success("Embedding & indexing complete!")
    else:
        st.warning("No embeddings found — possibly an empty PDF or no readable text.")

# Step 2: Ask Question
question = st.text_input("Ask a question about this PDF:")

if question and st.session_state["vector_store"] is not None:
    question_vec = get_embeddings([question], model="embedding-4o-latest")
    question_vec_np = np.array(question_vec, dtype=np.float32)

    results = st.session_state["vector_store"].search(question_vec_np, k=5)
    context_text = "\n".join([res[0] for res in results])

    prompt = f"""
    You are a helpful AI assistant. Only use the following PDF context to answer:
    {context_text}

    User Question: {question}
    """

    st.info("Generating final answer from AIMLAPI...")
    openai.api_key = st.secrets["AIMLAPI_KEY"]

    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an AI assistant that uses PDF context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
    answer = response.choices[0].message.content
    st.markdown(f"**Answer:** {answer}")

    with st.expander("Top Relevant Chunks"):
        for i, (chunk, dist) in enumerate(results):
            st.write(f"**Rank {i+1}** - Distance: {dist:.2f}")
            st.write(chunk[:300] + "...")
