from pathlib import Path
from app.models.document import DocumentMetadata

'''
Create data/raw/

Safely construct filename

Decode text

Save file

Calculate metadata

Return metadata
'''





def save_document(filename: str, contents: bytes) -> DocumentMetadata:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / Path(filename).name

    text = contents.decode("utf-8")

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    return DocumentMetadata(
        filename=file_path.name,
        file_type=file_path.suffix.lower(),
        size_bytes=len(contents),
        character_count=len(text),
        word_count=len(text.split()),
        status="uploaded",
    )