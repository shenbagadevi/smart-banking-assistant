import logging

from src.core.config import settings

logger = logging.getLogger(__name__)


def get_llm():
    """Return the configured chat model for the active LLM provider."""
    provider = (getattr(settings, "LLM_PROVIDER", "ollama") or "ollama").lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-ollama is required for the Ollama provider"
            ) from exc

        model_name = getattr(settings, "OLLAMA_MODEL", "gpt-5.5") or "gpt-5.5"
        logger.info("Initializing Ollama LLM provider with model=%s", model_name)
        return ChatOllama(model=model_name, temperature=0)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-openai is required for the OpenAI provider"
            ) from exc

        model_name = getattr(settings, "OPENAI_MODEL", None) or getattr(
            settings, "OPENAI_CHAT_MODEL", "gpt-5.5"
        )
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        logger.info("Initializing OpenAI LLM provider with model=%s", model_name)
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

    raise ValueError(f"Unsupported LLM provider: {provider}")
