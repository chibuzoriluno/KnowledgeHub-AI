from sentence_transformers import SentenceTransformer

from app.models.document import DocumentChunk


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text)
        return vector.tolist()

    def embed_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        chunk.embedding = self.embed_text(chunk.text)
        return chunk

    def embed_chunks(self, chunks: list[DocumentChunk],) -> list[DocumentChunk]:
        texts = [chunk.text for chunk in chunks]

        vectors = self.model.encode(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector.tolist()

        return chunks