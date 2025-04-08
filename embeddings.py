# embeddings.py
import openai
import streamlit as st
from typing import List

# No need for environment variables or load_dotenv
# We rely on st.secrets for the key

openai.api_base = "https://api.aimlapi.com/v1"

def get_embeddings(chunks: List[str], model: str = "embedding-4o-latest"):
    """
    Convert text chunks into vectors using an embedding model on AIMLAPI.
    """
    if not chunks:
        return []

    # Retrieve key from Streamlit secrets
    openai.api_key = st.secrets["AIMLAPI_KEY"]

    response = openai.Embedding.create(
        model=model,
        input=chunks
    )
    vectors = [item["embedding"] for item in response["data"]]
    return vectors
