from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8081,http://localhost:8080,http://localhost:3000,null"

    @property
    def cors_allow_all_dev(self) -> bool:
        return self.app_env == "development"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Generic LLM routing
    llm_backend: str = "anthropic"
    llm_model: str | None = None
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:11434"
    llm_timeout_sec: int = 120
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"

    # Evaluation judges
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

    # OpenAI (embeddings)
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "dharma-gpt"
    pinecone_environment: str = "us-east-1"

    # Vector DB backend
    vector_db_backend: str = "local"  # local | pinecone
    local_vector_index_name: str = "dharma-local"
    local_vector_namespace: str = "default"

    # Sarvam AI
    sarvam_api_key: str = ""
    indictrans2_model: str = "ai4bharat/indictrans2-indic-en-dist-200M"

    # STT backend: "sarvam" | "claude" | "indicwhisper" | "auto" (sarvam → claude → indicwhisper)
    stt_backend: str = "auto"
    indicwhisper_model: str = "openai/whisper-small"

    # Translation: when True, try local Ollama before cloud Anthropic in auto mode
    translation_local_first: bool = True

    # Admin / review API
    admin_api_key: str = ""
    admin_operator_api_key: str = ""
    staging_api_key: str = ""

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
