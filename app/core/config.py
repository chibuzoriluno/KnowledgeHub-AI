from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "KnowledgeHub AI"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = (
        "Production-ready Retrieval-Augmented Generation (RAG) API"
    )


settings = Settings()