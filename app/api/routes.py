from fastapi import APIRouter

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