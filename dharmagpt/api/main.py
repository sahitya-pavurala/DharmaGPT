from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from core.config import get_settings
from api.routes import query, audio, health, admin, feedback, chat

log = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("dharmagpt_api_start", env=settings.app_env, model=settings.anthropic_model)
    yield
    log.info("dharmagpt_api_shutdown")


app = FastAPI(
    title="DharmaGPT API",
    description="Dharmic wisdom from Hindu sacred texts — RAG-powered, citation-grounded",
    version="0.1.0",
    docs_url="/docs" if settings.app_env == "development" else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # In dev, allow all origins. In production, restrict via CORS_ORIGINS env var.
    allow_origins=["*"] if settings.app_env == "development" else settings.cors_origins_list,
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(admin.router, tags=["admin"])
app.include_router(chat.router, tags=["chat"])
