from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from uuid import uuid4

from src.api.v1.schemas.query_schema import QueryRequest
from src.api.v1.services.query_service import process_query
from src.api.v1.services.stream_service import stream_response

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

        return {
            "query": request.query,
            "answer": response.get("answer"),
            "query_path": response.get("query_path"),
            "document_name": response.get("document_name"),
            "page_no": response.get("page_no"),
            "policy_citations": response.get("policy_citations"),
            "sql_query_executed": response.get("sql_query_executed"),
            "retry_count": response.get("retry_count", 0),
            "confidence_score": response.get("confidence_score"),
            "correlation_id": correlation_id,
            "langsmith_trace_id": response.get("trace_id"),
        }

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
        response = process_query(
            query=request.query,
            user_id=request.user_id,
            correlation_id=correlation_id,
        )
        return StreamingResponse(
            stream_response(response.get("answer", "")),
            media_type="text/event-stream",
        )
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
