from src.api.v1.services.query_service import process_query, normalize_api_response


def test_hybrid_query():
    q = "What is my current balance and current home loan interest rate?"
    resp = process_query(q, user_id="test_user", correlation_id="hybrid_smoke")
    normalized = normalize_api_response(resp)

    assert "answer" in normalized and normalized["answer"].strip() != ""
    assert "query_path" in normalized
    # Hybrid should be indicated or SQL present
    assert normalized["query_path"] in ("HYBRID", "SQL", "RAG")
    # policy_citations should be present (possibly empty list)
    assert "policy_citations" in normalized
    # sql_query_executed should be present as a field
    assert "sql_query_executed" in normalized
