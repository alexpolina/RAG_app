# embeddings.py

import streamlit as st
from openai import OpenAI
from typing import List

def get_embeddings(chunks: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """
    Convert text chunks into embedding vectors using AIMLAPI's embedding endpoint.
    Uses a supported model name: text-embedding-ada-002.
    """
    if not chunks:
        return []

    # Initialize AIMLAPI client
    client = OpenAI(
        base_url="https://api.aimlapi.com/v1",
        api_key=st.secrets["TEXT_API_KEY"],
    )

    # Create embeddings
    response = client.embeddings.create(
        model=model,
        input=chunks,
    )

    # Return list of embedding vectors
    return [item["embedding"] for item in response.data]
