from src.api.v1.services.query_service import process_query, normalize_api_response


def test_api_contract_fields_present():
    resp = process_query("Hi", user_id="test_user", correlation_id="cid_123")
    meta = normalize_api_response(resp)
    # attach correlation and langsmith ids as the router would
    meta["correlation_id"] = "cid_123"
    meta["langsmith_trace_id"] = resp.get("trace_id") or ""

    required = [
        "answer",
        "query_path",
        "confidence_score",
        "retry_count",
        "correlation_id",
        "langsmith_trace_id",
        "policy_citations",
        "sql_query_executed",
    ]

    for k in required:
        assert k in meta
        assert meta[k] is not None
