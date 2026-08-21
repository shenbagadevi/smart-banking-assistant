from types import SimpleNamespace

from src.api.v1.agents.nodes import evaluate_answer_node as evaluate_node_module
from src.api.v1.agents.nodes import generate_answer_node as generate_node_module


def test_demo_mode_generate_answer_skips_openai(monkeypatch):
    monkeypatch.setattr(generate_node_module.settings, "DEMO_MODE", True, raising=False)

    def fail_llm():
        raise AssertionError("OpenAI should not be called when DEMO_MODE=true")

    monkeypatch.setattr(generate_node_module, "_get_llm", fail_llm)

    docs = [
        SimpleNamespace(
            page_content="The home loan interest rate is 7.2% for eligible borrowers.",
            metadata={
                "document_name": "northstar_home_loan_policy.pdf",
                "source_page": 5,
                "chunk_type": "text",
            },
        )
    ]

    state = {
        "query": "What is the home loan interest rate?",
        "route": "RAG",
        "retrieved_docs": docs,
    }

    result = generate_node_module.generate_answer_node(state)

    assert result["response"]["answer"]
    assert result["document_name"] == "northstar_home_loan_policy.pdf"
    assert result["page_no"] == "5"
    assert result["confidence_score"] == 0.72


def test_demo_mode_evaluation_skips_openai(monkeypatch):
    monkeypatch.setattr(evaluate_node_module.settings, "DEMO_MODE", True, raising=False)

    def fail_llm():
        raise AssertionError("OpenAI should not be called when DEMO_MODE=true")

    monkeypatch.setattr(evaluate_node_module, "_get_llm", fail_llm)

    state = {
        "query": "What is the home loan interest rate?",
        "route": "RAG",
        "response": {
            "answer": "The policy states the home loan interest rate is 7.2%."
        },
        "retrieved_docs": [
            SimpleNamespace(
                page_content="The home loan interest rate is 7.2%.",
                metadata={
                    "document_name": "northstar_home_loan_policy.pdf",
                    "source_page": 5,
                    "chunk_type": "text",
                },
            )
        ],
        "retry_count": 0,
    }

    result = evaluate_node_module.evaluate_answer_node(state)

    assert result["is_valid"] is True
    assert result["should_retry"] is False
    assert result["retry_count"] == 0
