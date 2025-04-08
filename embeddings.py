# embeddings.py
import openai
import os
from typing import List

# Configure AIMLAPI (OpenAI-compatible)
openai.api_base = "https://api.aimlapi.com/v1"
openai.api_key = os.getenv("AIMLAPI_KEY")  # Or set directly below

def get_embeddings(chunks: List[str], model: str = "embedding-4o-latest"):
    """
    Convert a list of text chunks into vectors using an embedding model from AIMLAPI.
    """
    if len(chunks) == 0:
        return []
    response = openai.Embedding.create(
        model=model,
        input=chunks
    )
    # Each item in response['data'] has an "embedding" field
    vectors = [item["embedding"] for item in response["data"]]
    return vectors
