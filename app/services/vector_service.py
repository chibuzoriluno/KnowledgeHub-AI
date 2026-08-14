import chromadb

from app.models.document import DocumentChunk


class VectorService:
    def __init__(self, path: str = "data/vector_db"):
        self.client = chromadb.PersistentClient(path=path)

        self.collection = self.client.get_or_create_collection(
            name="document_chunks"
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def search(self, query_embedding: list[float],top_k: int = 3,):
        return self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,)