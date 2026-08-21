from src.api.v1.services.query_service import process_query, normalize_api_response

queries = [
    ("What is the maximum tenure available for home loans?", "rag_test"),
    ("Show my last 3 transactions", "sql_test"),
    ("What is my loan balance and current home loan interest rate?", "hybrid_test"),
]

for q, cid in queries:
    print("\n--- QUERY:", q)
    resp = process_query(q, user_id="test_user", correlation_id=cid)
    print("RAW RESPONSE:", resp)
    print("NORMALIZED:", normalize_api_response(resp))
