from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from core.config import get_settings
from api.routes import query, audio, health

log = structlog.get_logger()
settings = get_settings()

app = FastAPI(
    title="DharmaGPT API",
    description="Dharmic wisdom from Hindu sacred texts — RAG-powered, citation-grounded",
    version="0.1.0",
    docs_url="/docs" if settings.app_env == "development" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])


@app.on_event("startup")
async def startup():
    log.info("dharmagpt_api_start", env=settings.app_env, model=settings.anthropic_model)
