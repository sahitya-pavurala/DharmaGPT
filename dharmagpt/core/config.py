from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8081,http://localhost:8080,http://localhost:3000,null"

    # Postgres (optional — leave empty to use SQLite)
    database_url: str = ""

    @property
    def cors_allow_all_dev(self) -> bool:
        return self.app_env == "development"

    # ── API Keys ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    sarvam_api_key: str = ""
    pinecone_api_key: str = ""

    # ── Backend selection (all pluggable via .env) ────────────────────────────
    # TRANSLATION_BACKEND: sarvam | anthropic | skip
    translation_backend: str = "sarvam"

    # EMBEDDING_BACKEND: openai
    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072

    # LLM_BACKEND: anthropic
    llm_backend: str = "anthropic"
    anthropic_model: str = "claude-sonnet-4-20250514"
    llm_model: str | None = None       # override model name if needed
    ollama_model: str = "qwen2.5:1.5b"  # compatibility alias for local test fixtures
    ollama_url: str = "http://localhost:11434"
    llm_timeout_sec: int = 120

    # RAG_BACKEND: pinecone | local
    rag_backend: str = "pinecone"
    vector_db_backend: str = "pinecone"  # legacy alias kept for backward compat
    pinecone_index_name: str = "dharma-gpt"
    pinecone_environment: str = "us-east-1"

    # local vector store (fallback / dev only)
    local_vector_index_name: str = "dharma-local"
    local_vector_namespace: str = "default"

    # STT_BACKEND: sarvam
    stt_backend: str = "sarvam"

    # ── RAG tuning ────────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_min_score: float = 0.35
    max_context_chars: int = 6000

    # ── Evaluation judges (Sarvam models understand Telugu/Sanskrit) ──────────
    evaluation_primary_backend: str = "openai"
    evaluation_primary_model: str = "sarvamai/sarvam-m"
    evaluation_primary_api_key: str = ""
    evaluation_primary_base_url: str = "http://localhost:8000/v1"
    evaluation_primary_timeout_sec: int = 120

    evaluation_secondary_backend: str = "openai"
    evaluation_secondary_model: str = "sarvamai/sarvam-30b"
    evaluation_secondary_api_key: str = ""
    evaluation_secondary_base_url: str = "http://localhost:8000/v1"
    evaluation_secondary_timeout_sec: int = 120

    # ── Admin / review API ────────────────────────────────────────────────────
    admin_api_key: str = ""
    admin_operator_api_key: str = ""
    staging_api_key: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        if self.llm_backend.lower() == "ollama":
            return self.ollama_model
        return self.anthropic_model

    def evaluation_model_for(self, role: str) -> tuple[str, str, str, str, int]:
        if role == "primary":
            return (
                self.evaluation_primary_backend,
                self.evaluation_primary_model,
                self.evaluation_primary_api_key,
                self.evaluation_primary_base_url,
                self.evaluation_primary_timeout_sec,
            )
        if role == "secondary":
            return (
                self.evaluation_secondary_backend,
                self.evaluation_secondary_model,
                self.evaluation_secondary_api_key,
                self.evaluation_secondary_base_url,
                self.evaluation_secondary_timeout_sec,
            )
        raise ValueError(f"Unknown evaluation role: {role}")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
