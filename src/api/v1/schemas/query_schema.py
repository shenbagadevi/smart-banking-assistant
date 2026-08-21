from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
