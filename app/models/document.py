from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    size_bytes: int
    character_count: int
    word_count: int
    status: str


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None