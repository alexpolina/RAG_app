# embeddings.py
import os
import openai  # We'll rely on openai-like client
from typing import List

# Setup client
openai.api_base = "https://api.aimlapi.com/v1"
openai.api_key = os.getenv("AIMLAPI_KEY")  # or pass directly

def get_embeddings(chunks: List[str], model: str = "embedding-4o-latest"):
    """
    Convert each chunk into a vector using the specified embedding model.
    """
    # create() can accept a list of strings
    response = openai.Embedding.create(
        model=model,
        input=chunks
    )
    # response['data'] is a list of dicts with "embedding"
    vectors = [item["embedding"] for item in response["data"]]
    return vectors
