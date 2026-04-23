from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8081"

    # Anthropic
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Generic LLM routing
    llm_backend: str = "anthropic"
    llm_model: str | None = None
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:11434"
    llm_timeout_sec: int = 120

    # OpenAI (embeddings)
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "dharma-gpt"
    pinecone_environment: str = "us-east-1"

    # Sarvam AI
    sarvam_api_key: str

    # Manual translation review API
    manual_translation_api_key: str = ""
    manual_translation_dataset_root: str = "knowledge/processed"
    manual_translation_audit_log: str = "knowledge/audit/manual_translation_audit.jsonl"
    manual_translation_allowed_datasets: str = ""

    # RAG
    rag_top_k: int = 5
    rag_min_score: float = 0.35
    max_context_chars: int = 6000

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or self.anthropic_model

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
