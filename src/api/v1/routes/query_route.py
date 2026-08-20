from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from uuid import uuid4

from src.api.v1.schemas.query_schema import QueryRequest
from src.api.v1.services.query_service import process_query
from src.api.v1.services.stream_service import stream_response
from src.api.v1.services.stream_service import request_cancel
from src.api.v1.services.query_service import normalize_api_response
import logging
import json

router = APIRouter(tags=["Smart Banking Assistant"])


@router.post("/query")
def query_assistant(request: QueryRequest):
    try:
        correlation_id = (
            request.correlation_id if request.correlation_id else str(uuid4())
        )
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        response = process_query(
            query=request.query,
            user_id=request.user_id,
            correlation_id=correlation_id,
        )

        logger = logging.getLogger(__name__)

        def generator():
            try:
                # stream answer tokens (SSE formatted)
                for token in stream_response(response.get("answer", "")):
                    yield token

                # indicate done and then emit metadata as SSE
                yield "data: [DONE]\n\n"

                metadata = normalize_api_response(response)
                metadata["correlation_id"] = (
                    correlation_id or metadata.get("correlation_id") or ""
                )
                metadata["langsmith_trace_id"] = (
                    response.get("trace_id") or metadata.get("langsmith_trace_id") or ""
                )
                metadata["query"] = request.query
                # include timing metrics when available
                if response.get("total_time") is not None:
                    metadata["total_time"] = response.get("total_time")

                # Emit metadata JSON as SSE data block
                yield f"data: {json.dumps(metadata)}\n\n"

            except Exception as e:
                logger.exception("Streaming generator failed: %s", e)
                yield f"data: [ERROR] Streaming failed: {str(e)}\n\n"

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(
            generator(), media_type="text/event-stream", headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
def query_assistant_stream(request: QueryRequest):
    try:
        correlation_id = (
            request.correlation_id if request.correlation_id else str(uuid4())
        )
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        # For streaming we still call process_query to initialize and then stream tokens
        # For streaming, invoke graph but stream tokens as they are generated.
        # process_query will synchronously start LangGraph which may produce
        # a partial answer (graph nodes may stream into the response object). We
        # then stream the textual answer tokens followed by a final metadata
        # JSON chunk so the UI can display sources.
        response = process_query(
            query=request.query,
            user_id=request.user_id,
            correlation_id=correlation_id,
        )

        logger = logging.getLogger(__name__)

        def generator():
            try:
                for token in stream_response(
                    response.get("answer", ""), correlation_id=correlation_id
                ):
                    yield token

                yield "data: [DONE]\n\n"

                metadata = normalize_api_response(response)
                metadata["correlation_id"] = (
                    correlation_id or metadata.get("correlation_id") or ""
                )
                metadata["langsmith_trace_id"] = (
                    response.get("trace_id") or metadata.get("langsmith_trace_id") or ""
                )
                metadata["query"] = request.query

                yield f"data: {json.dumps(metadata)}\n\n"
            except Exception as e:
                logger.exception("Streaming generator failed: %s", e)
                yield f"data: [ERROR] Streaming failed: {str(e)}\n\n"

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(
            generator(), media_type="text/event-stream", headers=headers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/cancel")
def cancel_query(correlation_id: str):
    try:
        if not correlation_id:
            raise HTTPException(status_code=400, detail="correlation_id required")
        request_cancel(correlation_id)
        return {"cancelled": True, "correlation_id": correlation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation/new")
def create_conversation(user_id: str):
    try:
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        return {"user_id": user_id, "correlation_id": str(uuid4())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
