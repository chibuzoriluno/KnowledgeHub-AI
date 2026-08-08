from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
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
Read its bytes
Create data/raw if necessary
Write file to disk
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

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / Path(file.filename).name

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    text = contents.decode("utf-8")

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    return {
        "filename": file_path.name,
        "size_bytes": len(contents),
        "character_count": len(text),
        "word_count": len(text.split()),
        "status": "uploaded",
    }