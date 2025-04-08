# vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add_embeddings(self, vectors: np.ndarray, text_chunks: list):
        self.index.add(vectors)
        self.chunks.extend(text_chunks)

    def search(self, query_vector: np.ndarray, k: int = 5):
        """
        Return top-k matches with (chunk, distance).
        """
        distances, indices = self.index.search(query_vector, k)
        results = []
        for i in range(k):
            chunk_idx = indices[0, i]
            distance = distances[0, i]
            if 0 <= chunk_idx < len(self.chunks):
                results.append((self.chunks[chunk_idx], distance))
        return results
