from src.api.v1.services.query_service import process_query


def test_chat_greeting_simple():
    resp = process_query("Hi", user_id="test_user", correlation_id="test_chat_1")
    assert resp is not None
    assert "answer" in resp
    assert "Hello" in resp["answer"]


def test_chat_greeting_with_name():
    resp = process_query(
        "Hi, I am Devi", user_id="test_user", correlation_id="test_chat_2"
    )
    assert resp is not None
    assert "answer" in resp
    assert "Hello Devi" in resp["answer"]


def test_chat_who_are_you():
    resp = process_query(
        "Who are you?", user_id="test_user", correlation_id="test_chat_3"
    )
    assert resp is not None
    assert "answer" in resp
    assert "NorthStar Bank" in resp["answer"]
