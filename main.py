# main.py
import streamlit as st
import numpy as np
import openai
import os

from pdf_loader import load_pdf_text, chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# Configure AIMLAPI
openai.api_base = "https://api.aimlapi.com/v1"
openai.api_key = os.getenv("AIMLAPI_KEY")

# Initialize or load a prebuilt FAISS index
@st.cache_resource
def init_vector_store(pdf_path: str):
    # 1. Load text
    pdf_text = load_pdf_text(pdf_path)
    # 2. Chunk
    chunks = chunk_text(pdf_text)
    # 3. Embed
    vectors = get_embeddings(chunks, model="embedding-4o-latest")
    vectors_np = np.array(vectors, dtype=np.float32)
    
    # 4. Setup VectorStore
    dimension = vectors_np.shape[1]  # e.g. 1536
    vs = VectorStore(dimension)
    vs.add_embeddings(vectors_np, chunks)
    return vs

# Load or build the vector store
vector_store = init_vector_store("my_lecture.pdf")

st.title("RAG System with AIMLAPI + Streamlit")

user_question = st.text_input("Ask a question about the PDF:")

if user_question:
    # 1. Embed the question
    question_vector = get_embeddings([user_question], model="embedding-4o-latest")
    question_vector_np = np.array(question_vector, dtype=np.float32)
    
    # 2. Search top chunks
    search_results = vector_store.search(question_vector_np, k=5)
    
    # 3. Combine top chunks into a single context
    context = "\n".join([res[0] for res in search_results])
    
    # 4. Send to LLM to generate final answer
    #    We'll use the `chat.completions.create` for your gpt-4o-mini
    #    or you could do something like:
    prompt = f"You have the following context from a PDF:\n{context}\n\nUser question: {user_question}\nGive a helpful answer that only uses the provided context."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",  # or your choice
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Only use the context provided for your answer."
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
    )

    final_answer = response.choices[0].message.content
    st.write(f"**Answer:** {final_answer}")

    # Optionally show the top chunks used
    with st.expander("Show Top Relevant Chunks"):
        for chunk, distance in search_results:
            st.markdown(f"- `{distance:.2f}`: {chunk[:200]} ...")
