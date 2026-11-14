"""STEP 2 - Turn chunks into vectors and store them in a FAISS index."""
import json

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, QUERY_PREFIX, INDEX_DIR


class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"   # use your GPU if possible
        self.model = SentenceTransformer(EMBED_MODEL, device=device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes every vector length 1, so
        # inner product == cosine similarity (that is what we want).
        return self.model.encode(
            texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype("float32")                                       # FAISS needs float32

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        return vec                                                # shape (1, dim)


def build_faiss_index(chunks: list[dict], embedder: Embedder) -> faiss.Index:
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])   # exact search using inner product
    index.add(vectors)                            # row i in the index == chunks[i]
    return index


def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks
