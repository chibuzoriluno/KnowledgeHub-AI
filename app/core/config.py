from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "KnowledgeHub AI"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = (
        "Production-ready Retrieval-Augmented Generation (RAG) API"
    )

    EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma2:2b"
    OLLAMA_TIMEOUT: float = 60.0

    RAG_DEFAULT_MAX_DISTANCE: float = 1.3


settings = Settings()