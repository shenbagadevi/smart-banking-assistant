from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()
# Application Configuration
PROJECT_NAME = "Smart Banking Assistant"
API_PREFIX = "/api/v1"

# Root directory for uploaded documents
UPLOAD_DIRECTORY = Path("data")

# Supported document types
ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx"}

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.5")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# PostgreSQL
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")


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

TEXT_CHUNK_SIZE = 1500

TEXT_CHUNK_OVERLAP = 300

EMBEDDING_DIMENSION = 1536

# Batch size for embedding requests
EMBEDDING_BATCH_SIZE = 50
