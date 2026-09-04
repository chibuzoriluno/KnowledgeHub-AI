import re

from app.models.document import DocumentChunk


def split_sentences(text: str) -> list[str]:
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_sentences = []

    for sentence in sentences:
        candidate_sentences = current_sentences + [sentence]
        candidate = " ".join(candidate_sentences)

        if not current_sentences:
            current_sentences = [sentence]
            continue

        if len(candidate) <= chunk_size:
            current_sentences.append(sentence)
            continue

        chunks.append(" ".join(current_sentences))

        overlap_sentences = []
        overlap_length = 0

        for previous_sentence in reversed(current_sentences):
            additional_length = len(previous_sentence)

            if overlap_sentences:
                additional_length += 1

            if overlap_length + additional_length > chunk_overlap:
                break

            candidate_with_overlap = (
                overlap_sentences
                + [sentence]
            )

            candidate_length = len(
                " ".join(candidate_with_overlap)
            )

            if candidate_length > chunk_size:
                break

            overlap_sentences.insert(
                0,
                previous_sentence,
            )

            overlap_length += additional_length

        current_sentences = overlap_sentences + [sentence]

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def build_chunks(
    text: str,
    document_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    text_chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for index, chunk in enumerate(text_chunks):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}_chunk_{index:03d}",
                document_id=document_id,
                chunk_index=index,
                text=chunk,
            )
        )

    return chunks