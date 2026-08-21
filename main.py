from fastapi import FastAPI

from src.api.v1.routes.document_routes import router as document_router
from src.api.v1.routes.query_route import router as query_router
from src.core.config import (
    API_PREFIX,
    PROJECT_NAME,
)
from src.core.logger import configure_logger

configure_logger()

app = FastAPI(
    title=PROJECT_NAME,
)

app.include_router(
    document_router,
    prefix=API_PREFIX,
)
app.include_router(
    query_router,
    prefix=API_PREFIX,
)
