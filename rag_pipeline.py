"""STEP 5 - The full pipeline: chunk -> embed -> index -> retrieve -> generate.

Usage:
  python rag_pipeline.py build              # read PDFs in data/, build the FAISS index
  python rag_pipeline.py ask "your question"
  python rag_pipeline.py chat               # interactive loop
"""
import argparse

from chunking import load_and_chunk
from embeddings import Embedder, build_faiss_index, save_index
from generation import generate_answer
from retrieval import FaissRetriever


def build():
    chunks = load_and_chunk()
    print(f"Created {len(chunks)} chunks")
    index = build_faiss_index(chunks, Embedder())
    save_index(index, chunks)
    print("Index saved to index/")


def answer(question: str, retriever) -> tuple[str, list[dict]]:
    hits = retriever.retrieve(question)          # R
    return generate_answer(question, hits), hits  # G


def show(question: str, retriever):
    text, hits = answer(question, retriever)
    print(f"\nANSWER:\n{text}\n\nSOURCES:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h['source']} p.{h['page']}  (score {h['score']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "ask", "chat"])
    ap.add_argument("question", nargs="?")
    args = ap.parse_args()

    if args.command == "build":
        build()
    elif args.command == "ask":
        show(args.question, FaissRetriever())
    else:
        r = FaissRetriever()
        while (q := input("\nQuestion (blank to quit): ").strip()):
            show(q, r)
