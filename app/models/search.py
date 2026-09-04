from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    max_distance: float | None = Field(default=None, gt=0)
    document_id: str | None = Field(default=None, min_length=1)


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    distance: float


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResult]


class RAGRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    max_distance: float | None = Field(default=None, gt=0)
    document_id: str | None = Field(default=None, min_length=1)


class RAGSource(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    distance: float


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: list[RAGSource]