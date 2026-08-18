import os
from typing import Any, Dict, List, Optional

import streamlit as st
import requests

DEFAULT_API_BASE_URL = os.getenv(
    "SMART_BANKING_API_BASE_URL",
    "http://localhost:8000/api/v1",
)

st.set_page_config(
    page_title="Smart Banking Assistant",
    page_icon="🏦",
    layout="wide",
)


def ensure_session_defaults() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo-user"
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = []


def get_api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)


def backend_status(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/docs", timeout=5)
        if response.ok:
            return True, "API reachable"
        return False, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


def create_conversation(user_id: str, base_url: str) -> Optional[str]:
    response = requests.post(
        f"{base_url.rstrip('/')}/conversation/new",
        params={"user_id": user_id},
        timeout=15,
    )
    if response.ok:
        payload = response.json()
        return payload.get("correlation_id")
    st.warning(f"Unable to start a conversation: {response.text}")
    return None


def upload_document(file_obj, base_url: str) -> Dict[str, Any]:
    files = {
        "file": (
            file_obj.name,
            file_obj.getvalue(),
            file_obj.type or "application/octet-stream",
        )
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/documents/upload",
        files=files,
        timeout=120,
    )
    payload = response.json() if response.content else {}
    if response.ok:
        return {"ok": True, "payload": payload}
    return {
        "ok": False,
        "payload": payload,
        "error": response.text,
    }


def ask_question(
    prompt: str, user_id: str, base_url: str, correlation_id: Optional[str]
) -> Dict[str, Any]:
    payload = {
        "query": prompt,
        "user_id": user_id,
        "correlation_id": correlation_id,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/query",
        json=payload,
        timeout=120,
    )
    if not response.ok:
        error_payload = response.json() if response.content else {}
        return {
            "ok": False,
            "error": error_payload.get("detail", response.text),
        }

    return {"ok": True, "payload": response.json()}


ensure_session_defaults()

st.title("🏦 Smart Banking Assistant")
st.caption("Demo frontend for the existing FastAPI backend and LangGraph orchestration")

with st.sidebar:
    st.header("Connection")
    st.text_input(
        "API Base URL",
        key="api_base_url",
        value=DEFAULT_API_BASE_URL,
        help="Use the FastAPI backend, for example http://localhost:8000/api/v1",
    )
    api_status, api_message = backend_status(get_api_base_url())
    st.status(
        "API reachable" if api_status else "API offline",
        state="complete" if api_status else "error",
    )
    st.caption(api_message)

    st.header("User")
    st.text_input("User ID", key="user_id")

    st.header("Conversation")
    if st.button("New conversation"):
        new_id = create_conversation(st.session_state.user_id, get_api_base_url())
        if new_id:
            st.session_state.conversation_id = new_id
            st.session_state.messages = []
            st.success(f"Conversation started with ID: {new_id}")

    if st.session_state.conversation_id:
        st.caption(f"Active correlation ID: {st.session_state.conversation_id}")
    else:
        st.caption("Start a new conversation to enable query tracking.")

    st.header("Knowledge Files")
    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "docx"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        with st.spinner("Uploading and ingesting..."):
            result = upload_document(uploaded, get_api_base_url())
        if result["ok"]:
            file_name = uploaded.name
            if file_name not in st.session_state.documents:
                st.session_state.documents.append(file_name)
            st.success(f"Uploaded: {file_name}")
            st.json(result["payload"])
        else:
            st.error(result.get("error", "Upload failed"))
            st.json(result.get("payload", {}))

    if st.session_state.documents:
        st.write("Recent uploads:")
        for name in st.session_state.documents:
            st.markdown(f"- {name}")

if not st.session_state.conversation_id:
    st.info("Create a new conversation from the sidebar to begin a banking chat.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("metadata"):
            with st.expander("Response details"):
                st.json(message["metadata"])

prompt = st.chat_input("Ask about banking products, policies, fees, or eligibility")
if prompt:
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = create_conversation(
            st.session_state.user_id,
            get_api_base_url(),
        )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            result = ask_question(
                prompt,
                st.session_state.user_id,
                get_api_base_url(),
                st.session_state.conversation_id,
            )

        if not result["ok"]:
            reply = f"The backend returned an error: {result['error']}"
            metadata = {"error": result["error"]}
            st.error(reply)
        else:
            payload = result["payload"]
            reply = payload.get("answer") or "No answer returned by the backend."
            metadata = {
                "query_path": payload.get("query_path"),
                "document_name": payload.get("document_name"),
                "page_no": payload.get("page_no"),
                "policy_citations": payload.get("policy_citations"),
                "sql_query_executed": payload.get("sql_query_executed"),
                "retry_count": payload.get("retry_count"),
                "confidence_score": payload.get("confidence_score"),
                "correlation_id": payload.get("correlation_id"),
                "langsmith_trace_id": payload.get("langsmith_trace_id"),
            }
            st.markdown(reply)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "metadata": metadata,
            }
        )

        if metadata:
            with st.expander("Response details"):
                st.json(metadata)
