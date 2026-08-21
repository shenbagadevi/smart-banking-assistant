from src.api.v1.services.stream_service import stream_response


def test_stream_response_format():
    tokens = list(stream_response("hello world this is a test"))
    # Ensure we produce SSE `data:` lines
    assert any(t.strip().startswith("data:") for t in tokens)
    # Ensure tokens present
    joined = "".join(tokens)
    assert "hello" in joined
    assert "test" in joined
