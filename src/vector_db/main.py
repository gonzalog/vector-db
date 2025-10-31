"""Main application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vector_db.core.config import settings
from vector_db.core.exceptions import (
    AlreadyExistsException,
    ConcurrencyException,
    NotFoundException,
    ValidationException,
    VectorDBException,
)
from vector_db.core.persistence.database import init_database, close_database
from vector_db.core.settings import Settings as PersistenceSettings
from vector_db.repositories.registry import initialize_repositories


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup: Initialize database
    persistence_settings = PersistenceSettings()
    await init_database(persistence_settings.DATABASE_PATH)

    # Initialize repositories (loads data from disk into memory)
    await initialize_repositories()

    yield

    # Shutdown: Close database connection
    await close_database()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routers
    register_routers(app)

    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers."""

    @app.exception_handler(NotFoundException)
    async def not_found_exception_handler(
        request: Request, exc: NotFoundException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": exc.message, "details": exc.details},
        )

    @app.exception_handler(AlreadyExistsException)
    async def already_exists_exception_handler(
        request: Request, exc: AlreadyExistsException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"message": exc.message, "details": exc.details},
        )

    @app.exception_handler(ValidationException)
    async def validation_exception_handler(
        request: Request, exc: ValidationException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"message": exc.message, "details": exc.details},
        )

    @app.exception_handler(ConcurrencyException)
    async def concurrency_exception_handler(
        request: Request, exc: ConcurrencyException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"message": exc.message, "details": exc.details},
        )

    @app.exception_handler(VectorDBException)
    async def vector_db_exception_handler(
        request: Request, exc: VectorDBException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": exc.message, "details": exc.details},
        )


def register_routers(app: FastAPI) -> None:
    """Register API routers."""
    from vector_db.api import chunks, documents, libraries

    app.include_router(libraries.router, prefix=settings.API_V1_PREFIX)
    app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
    app.include_router(chunks.router, prefix=settings.API_V1_PREFIX)


# Create the FastAPI app
app = create_app()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=settings.VERSION)
