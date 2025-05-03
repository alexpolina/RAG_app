# embeddings.py

import openai
import streamlit as st
from typing import List

# Configure AIMLAPI (OpenAI-compatible)
openai.api_base = "https://api.aimlapi.com/v1"

def get_embeddings(chunks: List[str], model: str = "embedding-4o-latest") -> List[List[float]]:
    """
    Convert a list of text chunks into embedding vectors using AIMLAPI's embedding endpoint.
    """
    if not chunks:
        return []

    # Retrieve API key from Streamlit Secrets
    openai.api_key = st.secrets["AIMLAPI_KEY"]

    # New v1.0+ interface for embeddings
    response = openai.embeddings.create(
        model=model,
        input=chunks
    )

    # Extract embeddings from the response
    vectors = [item["embedding"] for item in response["data"]]
    return vectors
