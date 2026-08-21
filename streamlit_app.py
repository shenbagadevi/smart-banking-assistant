import os
from typing import Any, Dict, List, Optional

import streamlit as st
import requests
import json

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
    # Use streaming request to receive tokens and trailing metadata
    try:
        # mark streaming state so UI can offer a Stop button
        st.session_state.is_streaming = True
        st.session_state.stop_requested = False

        with requests.post(
            f"{base_url.rstrip('/')}/query",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                error_payload = resp.json() if resp.content else {}
                return {"ok": False, "error": error_payload.get("detail", resp.text)}

            # Parse Server-Sent Events (SSE) `data:` lines. We accept tokens
            # as `data: <token>` and a final metadata JSON block as `data: { .. }`.
            text_parts = []
            metadata = {}
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        continue
                    # Error markers
                    if data.startswith("[ERROR]"):
                        metadata["error"] = data
                        break
                    # Metadata JSON block
                    if data.startswith("{") and data.endswith("}"):
                        try:
                            metadata = json.loads(data)
                        except Exception:
                            metadata = {}
                        break
                    # Otherwise it's a token piece
                    # allow frontend to request cancellation while streaming
                    if st.session_state.get("stop_requested"):
                        # attempt to notify backend to cancel
                        try:
                            requests.post(
                                f"{base_url.rstrip('/')}/query/cancel",
                                params={"correlation_id": correlation_id},
                                timeout=2,
                            )
                        except Exception:
                            pass
                        break
                    text_parts.append(data)

            answer_text = " ".join(text_parts).strip()
            # clear streaming flags
            st.session_state.is_streaming = False
            st.session_state.stop_requested = False
            payload_out = {"answer": answer_text, **(metadata or {})}
            return {"ok": True, "payload": payload_out}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


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
        # Provide a Stop button while streaming
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.session_state.get("is_streaming"):
                if st.button("Stop generating"):
                    # request cancellation; the streaming loop also checks stop_requested
                    st.session_state.stop_requested = True
                    try:
                        requests.post(
                            f"{get_api_base_url().rstrip('/')}/query/cancel",
                            params={"correlation_id": st.session_state.conversation_id},
                            timeout=2,
                        )
                    except Exception:
                        pass
        with col2:
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

            # Map API metadata to UI fields. Hide empty fields and prefer
            # normalized contract names from the API.
            metadata = {}
            qp = payload.get("query_path")
            if qp:
                metadata["query_path"] = qp
            doc = payload.get("document_name")
            if doc and doc != "NA":
                metadata["document_name"] = doc
            # policy_citations may be a list of objects
            cites = payload.get("policy_citations") or []
            if cites:
                # Show only document and section where present
                first = cites[0]
                if isinstance(first, dict):
                    docname = first.get("document")
                    section = first.get("section")
                    if docname:
                        metadata.setdefault("sources", []).append(docname)
                    if section:
                        metadata.setdefault("sections", []).append(section)

            sqlq = payload.get("sql_query_executed")
            if sqlq and sqlq != "NA":
                metadata["sql_query_executed"] = sqlq

            metadata["retry_count"] = payload.get("retry_count")
            metadata["confidence_score"] = payload.get("confidence_score")
            metadata["correlation_id"] = payload.get("correlation_id")
            metadata["langsmith_trace_id"] = payload.get("langsmith_trace_id")

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
