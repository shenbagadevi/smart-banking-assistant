from pathlib import Path
import os
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()
# Application Configuration
PROJECT_NAME = "Smart Banking Assistant"
API_PREFIX = "/api/v1"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# Root directory for uploaded documents
UPLOAD_DIRECTORY = Path("data")
IMAGE_DIRECTORY = Path("data/images")


# Supported document types
ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx"}

# LLM provider selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-5.5")

# Embedding provider selection
EMBEDDING_PROVIDER = (
    os.getenv("EMBEDDING_PROVIDER")
    or ("ollama" if LLM_PROVIDER == "ollama" else "openai")
).lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "text-embedding-3-small")

# OpenAI (kept for future use)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or OPENAI_CHAT_MODEL
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# Cohere
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# LangSmith
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "smart-banking-assistant")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() in (
    "1",
    "true",
    "yes",
)

# PostgreSQL (canonical configuration source)
PG_CONNECTION_STRING = (
    os.getenv("PG_CONNECTION_STRING")
    or os.getenv("PG_RDBMS_CONNECTION_STRING")
    or os.getenv("PG_RDBMS_CONNECTION")
)
PG_VECTOR_CONNECTION = (
    os.getenv("PG_VECTOR_CONNECTION")
    or os.getenv("PG_VECTOR_CONNECTION_STRING")
    or PG_CONNECTION_STRING
)

GUARDRAIL_ENABLED = os.getenv("GUARDRAIL_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
PII_DETECTION_ENABLED = os.getenv("PII_DETECTION_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
PROMPT_INJECTION_ENABLED = os.getenv("PROMPT_INJECTION_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
CITATION_CHECK_ENABLED = os.getenv("CITATION_CHECK_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
HALLUCINATION_CHECK_ENABLED = os.getenv(
    "HALLUCINATION_CHECK_ENABLED", "true"
).lower() in (
    "1",
    "true",
    "yes",
)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
STREAM_CANCELLATION_ENABLED = os.getenv(
    "STREAM_CANCELLATION_ENABLED", "true"
).lower() in (
    "1",
    "true",
    "yes",
)
MEMORY_SAVE_ON_SUCCESS_ONLY = os.getenv(
    "MEMORY_SAVE_ON_SUCCESS_ONLY", "true"
).lower() in (
    "1",
    "true",
    "yes",
)

settings = SimpleNamespace(
    COHERE_API_KEY=COHERE_API_KEY,
    LANGCHAIN_API_KEY=LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT=LANGCHAIN_PROJECT,
    DEMO_MODE=DEMO_MODE,
    LLM_PROVIDER=LLM_PROVIDER,
    OLLAMA_MODEL=OLLAMA_MODEL,
    EMBEDDING_PROVIDER=EMBEDDING_PROVIDER,
    HF_EMBEDDING_MODEL=HF_EMBEDDING_MODEL,
    OPENAI_API_KEY=OPENAI_API_KEY,
    OPENAI_MODEL=OPENAI_MODEL,
    OPENAI_CHAT_MODEL=OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL=OPENAI_EMBEDDING_MODEL,
    PG_CONNECTION_STRING=PG_CONNECTION_STRING,
    PG_VECTOR_CONNECTION=PG_VECTOR_CONNECTION,
    GUARDRAIL_ENABLED=GUARDRAIL_ENABLED,
    PII_DETECTION_ENABLED=PII_DETECTION_ENABLED,
    PROMPT_INJECTION_ENABLED=PROMPT_INJECTION_ENABLED,
    CITATION_CHECK_ENABLED=CITATION_CHECK_ENABLED,
    HALLUCINATION_CHECK_ENABLED=HALLUCINATION_CHECK_ENABLED,
    CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD,
    STREAM_CANCELLATION_ENABLED=STREAM_CANCELLATION_ENABLED,
    MEMORY_SAVE_ON_SUCCESS_ONLY=MEMORY_SAVE_ON_SUCCESS_ONLY,
)


VISION_PROMPT = """
            Analyze this document image and describe all visible information accurately.

            Include:
            - Title or heading
            - Charts, graphs, tables, figures, and diagrams
            - Labels, legends, axes, and units
            - Key values, numbers, percentages, and dates
            - Important entities, terms, and abbreviations
            - Any conclusions or notable observations

            Return a factual, structured description optimized for semantic search. 
            Do not omit visible information or make assumptions.
"""

EMBEDDING_DIMENSION = 1536

# Chunk splitter settings aligned to the ingestion strategy.
TEXT_CHUNK_SIZE = 512

TEXT_CHUNK_OVERLAP = 100

# Batch size for embedding requests
EMBEDDING_BATCH_SIZE = 50

# DOCX conversion config: 'auto' tries docx2pdf then soffice, 'docx2pdf', 'soffice', or 'none'
DOCX_CONVERTER = os.getenv("DOCX_CONVERTER", "auto")
# Optional explicit path to soffice (LibreOffice)
SOFFICE_PATH = os.getenv("SOFFICE_PATH")
# When true, use a lighter Docling pipeline that skips heavy layout/ocr models.
# Set DOCLING_LIGHT_MODE=true in .env to enable (useful on machines without compilers/GPU).
DOCLING_LIGHT_MODE = os.getenv("DOCLING_LIGHT_MODE", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Whether tables should inherit nearby heading/section metadata. Set to
# "false" in env to prevent attaching headings to table elements.
TABLES_INHERIT_HEADINGS = os.getenv("TABLES_INHERIT_HEADINGS", "true").lower() in (
    "1",
    "true",
    "yes",
)
