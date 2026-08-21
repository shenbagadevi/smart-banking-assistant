import sys
import os

# Ensure project root is on sys.path so tests can be imported when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_chat_route import (
    test_chat_greeting_simple,
    test_chat_greeting_with_name,
    test_chat_who_are_you,
)
from tests.test_stream_response import test_stream_response_format
from tests.test_api_contract import test_api_contract_fields_present
from tests.test_hybrid_route import test_hybrid_query


def run_all():
    tests = [
        test_chat_greeting_simple,
        test_chat_greeting_with_name,
        test_chat_who_are_you,
        test_stream_response_format,
        test_api_contract_fields_present,
        test_hybrid_query,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print("PASS:", t.__name__)
        except AssertionError as e:
            print("FAIL:", t.__name__, e)
            failures.append((t.__name__, e))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
