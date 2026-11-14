"""STEP 1 - Load PDFs with PyMuPDF and split them into recursive chunks."""
import sys
from pathlib import Path

import fitz  # PyMuPDF is imported under the name "fitz"
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR #from config.py, loads the chunk size, overlap and data(PDF)


def load_pdf(path: Path) -> list[dict]:
    """Return one dict per page: {"text", "page", "source"}."""
    pages = []
    with fitz.open(path) as doc:                  # open the PDF; 'with' closes it afterwards
        for page_number, page in enumerate(doc, start=1):   # page numbers start at 1
            text = page.get_text().strip()        # extract the plain text of this page
            if text:                              # skip blank / image-only pages
                pages.append({"text": text, "page": page_number, "source": path.name})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Recursive chunking: try to split on paragraphs first, then lines, then sentences,
    then words, and only as a last resort on single characters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],   # tried in this order
    )
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):   # split each page separately
            chunks.append({
                "chunk_id": len(chunks),               # running number 0,1,2,...
                "text": piece,
                "page": p["page"],                     # keep metadata for citations
                "source": p["source"],
            })
    return chunks


def load_and_chunk(path: Path | None = None) -> list[dict]:
    """Chunk one PDF, or every PDF inside the data/ folder."""
    pdfs = [Path(path)] if path else sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}. Put some there first.")
    pages = []
    for pdf in pdfs:
        pages.extend(load_pdf(pdf))
    return chunk_pages(pages)


if __name__ == "__main__":
    chunks = load_and_chunk(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(f"{len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n--- chunk {c['chunk_id']} | {c['source']} p.{c['page']} ---\n{c['text']}")
