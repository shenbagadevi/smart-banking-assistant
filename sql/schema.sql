-- ============================================================
-- Smart Banking Assistant
-- Multimodal RAG Knowledge Base
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 1. Knowledge Documents
-- One record per uploaded document
-- ============================================================

CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    document_name   TEXT NOT NULL UNIQUE,

    source_path     TEXT NOT NULL,

    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 2. Knowledge Chunks
-- Stores text, table and image chunks
-- ============================================================

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id              BIGSERIAL PRIMARY KEY,

    chunk_id        UUID NOT NULL UNIQUE,

    document_id     UUID NOT NULL
                    REFERENCES knowledge_documents(document_id)
                    ON DELETE CASCADE,

    document_name   TEXT NOT NULL,

    chunk_type      TEXT NOT NULL
                    CHECK (
                        chunk_type IN (
                            'text',
                            'table',
                            'image',
                            'image_caption'
                        )
                    ),

    content         TEXT NOT NULL,

    source_page     INTEGER,

    section         TEXT,

    embedding       VECTOR(1536),

    metadata        JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 3. Vector Index
-- Used later by LangGraph RAG retrieval
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
ON knowledge_chunks
USING hnsw (embedding vector_cosine_ops);


-- ============================================================
-- 4. Document Index
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document
ON knowledge_chunks (document_id);


-- ============================================================
-- 5. Chunk Type Index
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_type
ON knowledge_chunks (chunk_type);


-- ============================================================
-- 6. Page Index
-- Useful for citations and document navigation
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source_page
ON knowledge_chunks (source_page);

ALTER TABLE knowledge_chunks
ADD COLUMN image_path TEXT;