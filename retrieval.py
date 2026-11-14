"""STEP 3 - The R in RAG: find the chunks most similar to the question."""
from config import TOP_K
from embeddings import Embedder, load_index


class FaissRetriever:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index, self.chunks = load_index()

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        q = self.embedder.embed_query(question)
        scores, ids = self.index.search(q, k)       # both have shape (1, k)
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:                             # FAISS returns -1 if fewer than k vectors exist
                continue
            c = self.chunks[i]
            hits.append({**c, "score": float(score)})
        return hits                                 # already sorted best-first
