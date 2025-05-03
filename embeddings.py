# embeddings.py

import streamlit as st
from openai import OpenAI
from typing import List

# Fallback import; will succeed once sentence-transformers is installed
from sentence_transformers import SentenceTransformer
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embeddings(chunks: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """
    Try AIMLAPI first; on quota error, fall back to a local sentence-transformers model.
    """
    if not chunks:
        return []

    # First, try hosted embeddings
    try:
        client = OpenAI(
            base_url="https://api.aimlapi.com/v1",
            api_key=st.secrets["TEXT_API_KEY"],
        )
        resp = client.embeddings.create(model=model, input=chunks)
        return [item.embedding for item in resp.data]

    except Exception as e:
        # If any error (e.g., quota), fall back locally
        st.warning(f"AIMLAPI embeddings failed ({e}), switching to local model.")
        local_encoder = SentenceTransformer(LOCAL_MODEL_NAME)
        return local_encoder.encode(chunks).tolist()
