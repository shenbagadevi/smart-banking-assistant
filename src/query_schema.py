from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="User banking query")
    session_id: Optional[str] = None


class AIResponse(BaseModel):
    answer: str
    policy_citations: str
    page_no: str
    document_name: str
    query_path: Optional[str] = None
    sql_query_executed: Optional[str] = None
    retry_count: Optional[int] = 0
