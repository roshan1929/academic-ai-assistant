"""FAISS-backed vector store placeholder."""


class FaissStore:
    """Minimal placeholder interface for a vector store."""

    def __init__(self, store_path: str = None):
        self.store_path = store_path

    def add_vectors(self, vectors):
        """Placeholder for adding vectors to the store."""
        pass

    def search(self, query_vector, top_k: int = 5):
        """Placeholder for searching the store; returns empty list."""
        return []
