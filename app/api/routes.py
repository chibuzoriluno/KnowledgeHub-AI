import logging
import time
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.document_service import save_document
from app.services.generation_service import GenerationServiceError
from app.models.search import (
    SearchRequest,
    SearchResponse,
    RAGRequest,
    RAGResponse,
)
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
def root():
    return {
        "message": "Welcome to KnowledgeHub AI!",
        "status": "running",
    }


@router.get("/health")
def health():
    return {
        "status": "healthy",
    }


# upload end point
'''
Uploadfile =
Receive uploaded file
File validation
Return confirmation & metadata
'''


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if Path(file.filename).suffix.lower() != ".txt":
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are currently supported.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    logger.info(
        "Document upload accepted: filename=%s size_bytes=%d",
        file.filename,
        len(contents),
    )

    try:
        metadata, chunks = save_document(
            file.filename,
            contents,
        )
    except UnicodeDecodeError:
        logger.warning(
            "Document upload rejected: invalid UTF-8 filename=%s",
            file.filename,
        )
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be valid UTF-8 text.",
        )

    return {
        **metadata.model_dump(),
        "chunk_count": len(chunks),
        }



retrieval_service = RetrievalService()
rag_service = RAGService()

@router.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    results = retrieval_service.search(
        request.query,
        top_k=request.top_k,
        max_distance=request.max_distance,
        document_id=request.document_id,
    )

    logger.info(
        "Semantic search completed: top_k=%d result_count=%d",
        request.top_k,
        results.result_count,
        )

    return results


@router.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest):
    start_time = time.perf_counter()

    logger.info(
        "RAG request started: top_k=%d",
        request.top_k,
    )

    try:
        result = await rag_service.answer(
            query=request.query,
            top_k=request.top_k,
            max_distance=request.max_distance,
            document_id=request.document_id,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "RAG request completed: source_count=%d duration_ms=%.2f",
            len(result.sources),
            duration_ms,
        )

        return result

    except GenerationServiceError as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "RAG generation failed: duration_ms=%.2f error=%s",
            duration_ms,
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc