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
    # TRANSLATION_BACKEND: sarvam | anthropic | openai | ollama | indictrans2 | skip
    translation_backend: str = "sarvam"
    indictrans2_model: str = "ai4bharat/indictrans2-indic-en-dist-200M"

    # EMBEDDING_BACKEND: openai
    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072

    # LLM_BACKEND: anthropic | openai | ollama
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

    # STT_BACKEND: sarvam | indicconformer
    stt_backend: str = "sarvam"

    # ── RAG tuning ────────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_min_score: float = 0.35
    max_context_chars: int = 6000

    # ── Evaluation judge ──────────────────────────────────────────────────────
    # Default policy: one Anthropic judge call per response. Local Ollama is used
    # for serving; Sarvam is reserved for corpus creation/transcription.
    evaluation_primary_backend: str = "anthropic"
    evaluation_primary_model: str = "claude-sonnet-4-20250514"
    evaluation_primary_api_key: str = ""
    evaluation_primary_base_url: str = ""
    evaluation_primary_timeout_sec: int = 120

    # Retained for compatibility with older reports/configs; current scorer uses
    # only the primary judge by default.
    evaluation_secondary_backend: str = "anthropic"
    evaluation_secondary_model: str = "claude-sonnet-4-20250514"
    evaluation_secondary_api_key: str = ""
    evaluation_secondary_base_url: str = ""
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
        if self.llm_backend.lower() == "openai":
            return "gpt-4.1-mini"
        return self.anthropic_model

    def evaluation_model_for(self, role: str) -> tuple[str, str, str, str, int]:
        def _fallback_key(backend: str, configured_key: str) -> str:
            if configured_key:
                return configured_key
            if backend == "anthropic":
                return self.anthropic_api_key
            if backend == "openai":
                return self.openai_api_key
            return ""

        def _fallback_base_url(backend: str, configured_url: str) -> str:
            if configured_url:
                return configured_url
            if backend == "ollama":
                return self.ollama_url
            return ""

        if role == "primary":
            backend = self.evaluation_primary_backend
            return (
                backend,
                self.evaluation_primary_model,
                _fallback_key(backend, self.evaluation_primary_api_key),
                _fallback_base_url(backend, self.evaluation_primary_base_url),
                self.evaluation_primary_timeout_sec,
            )
        if role == "secondary":
            backend = self.evaluation_secondary_backend
            return (
                backend,
                self.evaluation_secondary_model,
                _fallback_key(backend, self.evaluation_secondary_api_key),
                _fallback_base_url(backend, self.evaluation_secondary_base_url),
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
