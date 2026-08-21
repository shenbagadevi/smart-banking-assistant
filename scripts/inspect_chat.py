from src.api.v1.services.query_service import process_query

resp = process_query("Who are you?", user_id="test_user", correlation_id="inspect_1")
print(resp)
print("ANSWER:", resp.get("answer"))
