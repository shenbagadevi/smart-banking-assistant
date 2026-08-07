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
