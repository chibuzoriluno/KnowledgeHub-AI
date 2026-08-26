from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "KnowledgeHub AI"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = (
        "Production-ready Retrieval-Augmented Generation (RAG) API"
    )
    RAG_DEFAULT_MAX_DISTANCE: float = 1.3


settings = Settings()