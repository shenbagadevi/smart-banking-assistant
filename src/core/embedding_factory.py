import logging

from src.core.config import settings

logger = logging.getLogger(__name__)


def get_embeddings():
    """Return the configured embeddings provider without hard-coding OpenAI."""
    provider = (getattr(settings, "EMBEDDING_PROVIDER", "openai") or "openai").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model_name = getattr(
            settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        logger.info("Initializing OpenAI embeddings provider with model=%s", model_name)
        return OpenAIEmbeddings(model=model_name, api_key=api_key)

    if provider == "ollama":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-huggingface is required for the local Ollama/HF embedding provider"
            ) from exc

        model_name = getattr(settings, "HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        logger.info(
            "Initializing HuggingFace embeddings provider with model=%s", model_name
        )
        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(f"Unsupported embedding provider: {provider}")
