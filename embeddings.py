# embeddings.py

import streamlit as st
from openai import OpenAI
from typing import List

# Add import for local fallback
from sentence_transformers import SentenceTransformer

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embeddings(chunks: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """
    Try AIMLAPI first; on quota error, fall back to a local sentence-transformers model.
    """
    if not chunks:
        return []

    try:
        # Attempt hosted embeddings
        client = OpenAI(
            base_url="https://api.aimlapi.com/v1",
            api_key=st.secrets["TEXT_API_KEY"],
        )
        resp = client.embeddings.create(model=model, input=chunks)
        return [item.embedding for item in resp.data]

    except PermissionDeniedError:
        st.warning("AIMLAPI quota reached — switching to local embeddings (sentence-transformers).")
        # Local fallback
        local_encoder = SentenceTransformer(LOCAL_MODEL_NAME)
        return local_encoder.encode(chunks).tolist()
