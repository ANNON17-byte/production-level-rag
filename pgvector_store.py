"""STEP 7 - Same retrieval, but the vectors live in Postgres (pgvector) on Supabase.

Usage:
  python pgvector_store.py ingest      # chunk PDFs, embed, insert into Postgres
"""
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from config import DATABASE_URL, EMBED_DIM, TOP_K
from embeddings import Embedder


def connect():
    if not DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL in your .env file")
    # prepare_threshold=None -> avoids prepared-statement errors behind Supabase's pooler
    conn = psycopg.connect(DATABASE_URL, autocommit=True, prepare_threshold=None)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")   # must exist BEFORE register_vector
    register_vector(conn)                                   # lets us pass numpy arrays as vectors
    return conn


def init_schema(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            page INT,
            embedding VECTOR({EMBED_DIM}) NOT NULL
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                 "ON chunks USING hnsw (embedding vector_cosine_ops)")


def insert_chunks(conn, chunks: list[dict], vectors):
    rows = [(c["text"], c["source"], c["page"], v) for c, v in zip(chunks, vectors)]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (content, source, page, embedding) VALUES (%s, %s, %s, %s)", rows
        )


class PgRetriever:
    """Drop-in replacement for FaissRetriever: same .retrieve() method."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.conn = connect()

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        q = self.embedder.embed_query(question)[0]
        # <=> is cosine DISTANCE (0 = identical), so score = 1 - distance
        sql = """SELECT id, content AS text, source, page, 1 - (embedding <=> %s) AS score
                 FROM chunks ORDER BY embedding <=> %s LIMIT %s"""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (q, q, k))
            rows = cur.fetchall()
        for r in rows:
            r["score"] = float(r["score"])
        return rows


if __name__ == "__main__":
    from chunking import load_and_chunk
    chunks = load_and_chunk()
    vectors = Embedder().embed_documents([c["text"] for c in chunks])
    conn = connect()
    init_schema(conn)
    insert_chunks(conn, chunks, vectors)
    print(f"Inserted {len(chunks)} chunks into Postgres")
