from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    size_bytes: int
    character_count: int
    word_count: int
    status: str