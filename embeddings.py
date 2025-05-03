# embeddings.py

import openai
import streamlit as st
from typing import List

# Point at the AIMLAPI base URL (OpenAI-compatible)
openai.api_base = "https://api.aimlapi.com/v1"

def get_embeddings(chunks: List[str], model: str = "embedding-4o-latest") -> List[List[float]]:
    """
    Convert text chunks into embedding vectors using AIMLAPI's embedding endpoint.
    """
    if not chunks:
        return []

    # Pull your key from the secret named "TEXT_API_KEY"
    openai.api_key = st.secrets["TEXT_API_KEY"]

    # Use the v1.0+ interface for embeddings
    response = openai.embeddings.create(
        model=model,
        input=chunks
    )

    return [item["embedding"] for item in response["data"]]
