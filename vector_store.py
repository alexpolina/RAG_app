# vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int):
        # dimension must match embedding dimension from your model
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add_embeddings(self, vectors: np.ndarray, text_chunks: list):
        self.index.add(vectors)
        self.chunks.extend(text_chunks)

    def search(self, query_vector: np.ndarray, k: int = 5):
        """
        Return the top-k similar chunks based on L2 distance
        """
        distances, indices = self.index.search(query_vector, k)
        # Return chunk strings for top results
        return [(self.chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
