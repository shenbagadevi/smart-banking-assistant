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

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.5")
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

settings = SimpleNamespace(
    COHERE_API_KEY=COHERE_API_KEY,
    LANGCHAIN_API_KEY=LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT=LANGCHAIN_PROJECT,
    DEMO_MODE=DEMO_MODE,
    PG_CONNECTION_STRING=PG_CONNECTION_STRING,
    PG_VECTOR_CONNECTION=PG_VECTOR_CONNECTION,
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
