import importlib


def test_factory_uses_ollama_by_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import src.core.config as config

    importlib.reload(config)

    import src.core.llm_factory as llm_factory

    importlib.reload(llm_factory)

    llm = llm_factory.get_llm()
    assert llm is not None
    assert llm.model == "gpt-5.5"


def test_factory_uses_openai_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import src.core.config as config

    importlib.reload(config)

    import src.core.llm_factory as llm_factory

    importlib.reload(llm_factory)

    llm = llm_factory.get_llm()
    assert llm is not None
    assert llm.model == "gpt-4o-mini"
