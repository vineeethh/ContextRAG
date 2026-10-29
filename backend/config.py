from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Only the direct-OpenAI LLM path (LLM_PROVIDER=openai) needs this; embeddings
    # and the default LLM go through OpenRouter.
    openai_api_key: str = ""
    cohere_api_key: str = ""
    github_token: str = ""

    # Tenant-uploaded PDF / Markdown files. In Docker this is a volume shared
    # by the ingestion service (writes) and the Celery worker (reads).
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 20

    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql+asyncpg://kapa:kapa_dev_password@localhost:5432/kapa_rag"

    # Served through OpenRouter, so the model id carries the vendor prefix.
    openai_embedding_model: str = "openai/text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    llm_default_model: str = "gpt-4o"
    llm_fast_model: str = "gpt-4o-mini"
    llm_fast_token_threshold: int = 500

    llm_provider: str = "openai"
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    # Thinking tokens bill as output and count against max_tokens, so it is off
    # unless a run explicitly wants it.
    openrouter_reasoning: bool = False

    max_context_tokens: int = 6000
    rate_limit_per_minute: int = 20

    # Coverage gaps: a query whose top Cohere rerank score is below this
    # threshold is logged as a question the knowledge base can't answer.
    coverage_gap_threshold: float = 0.3

    service_port: int = 8000

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_access_expire_minutes: int = 15
    jwt_refresh_expire_days: int = 30

    # Observability — Langfuse tracing is enabled only when both keys are set
    environment: str = "dev"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
