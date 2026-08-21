import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.v1.services.query_service import process_query
import json

tests = [
    ("Hello", "CHAT"),
    ("Thank you", "CHAT"),
    ("What can you help me with?", "CHAT"),
    ("What is my account balance?", "SQL"),
    ("What is home loan interest rate?", "RAG"),
]

for q, expected in tests:
    print("\n--- TEST QUERY: {} | expected route={} ---".format(q, expected))
    try:
        resp = process_query(q, user_id="test-user", correlation_id="test-thread")
        route = resp.get("query_path") or resp.get("route") or ""
        route = route.upper()
        print(json.dumps(resp, indent=2))
        if route == expected:
            print("PASS: route=", route)
        else:
            print(f"FAIL: expected {expected} but got {route}")
    except Exception as e:
        print("Error running query:", e)
