from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import structlog

from core.config import get_settings
from api.routes import query, audio, health, admin, feedback

log = structlog.get_logger()
settings = get_settings()
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
STATIC_DIR = WEB_DIR / "static"
INDEX_FILE = WEB_DIR / "index.html"


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
    # In dev, allow all origins so the local HTML file (file://) can call the API.
    # In production, restrict to the deployed frontend origin via CORS_ORIGINS env var.
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

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False, response_model=None)
async def root():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"name": "DharmaGPT Beta Server", "status_url": "/health", "api": "/api/v1/query"}
