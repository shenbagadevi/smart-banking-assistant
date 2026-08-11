from fastapi import APIRouter, HTTPException

from src.api.v1.schemas.query_schema import QueryRequest
from src.api.v1.services.query_service import process_query


router = APIRouter(prefix="/api/v1", tags=["Smart Banking Assistant"])


@router.post("/query")
def query_assistant(request: QueryRequest):
    try:
        print("====== INSIDE query_assistant route ======")

        response = process_query(request.query)

        return {
            "query": request.query,
            "answer": response.get("answer"),
            "query_path": response.get("query_path"),
            "document_name": response.get("document_name"),
            "page_no": response.get("page_no"),
            "policy_citations": response.get("policy_citations"),
            "sql_query_executed": response.get("sql_query_executed"),
            "retry_count": response.get("retry_count", 0),
        }

    except Exception as e:
        print("Error in query_assistant route:", e)
        raise HTTPException(status_code=500, detail=str(e))
