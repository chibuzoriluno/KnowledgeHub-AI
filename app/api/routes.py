from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.document_service import save_document
from app.models.search import SearchRequest, SearchResponse
from app.services.retrieval_service import RetrievalService

router = APIRouter()


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

    metadata, chunks = save_document(file.filename, contents)

    return {
        **metadata.model_dump(),
        "chunk_count": len(chunks),
        }



retrieval_service = RetrievalService()

@router.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    results = retrieval_service.search(
        request.query,
        top_k=request.top_k,
        max_distance=request.max_distance,
        )

    return results