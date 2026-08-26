from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import settings
from app.models.document import DocumentChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingService:
    def __init__(
        self,
        model_name: str | None = None,
    ):
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL

        self.model = get_embedding_model(model_name)

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text)
        return vector.tolist()

    def embed_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        chunk.embedding = self.embed_text(chunk.text)
        return chunk

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        texts = [chunk.text for chunk in chunks]

        vectors = self.model.encode(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector.tolist()

        return chunks