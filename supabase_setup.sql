-- Run this once in Supabase: Dashboard -> SQL Editor -> New query -> Run
create extension if not exists vector;

create table if not exists chunks (
    id         bigserial primary key,
    content    text        not null,
    source     text,
    page       int,
    embedding  vector(384) not null       -- 384 = output size of bge-small-en-v1.5
);

-- Fast approximate nearest-neighbour index (cosine distance)
create index if not exists chunks_embedding_idx
    on chunks using hnsw (embedding vector_cosine_ops);
