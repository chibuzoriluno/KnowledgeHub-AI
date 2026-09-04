import pytest

from app.services.chunk_service import build_chunks, chunk_text


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than zero",
    ):
        chunk_text("some text", chunk_size=0)


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative",
    ):
        chunk_text("some text", chunk_size=10, chunk_overlap=-1)


def test_chunk_text_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        chunk_text("some text", chunk_size=10, chunk_overlap=10)


def test_chunk_text_returns_non_empty_chunks():
    chunks = chunk_text(
        "abcdefghijklmnopqrstuvwxyz",
        chunk_size=10,
        chunk_overlap=2,
    )

    assert chunks
    assert all(chunk for chunk in chunks)


def test_chunk_text_applies_sentence_overlap():
    text = (
        "First sentence is here. "
        "Second sentence contains several important words. "
        "Third sentence is here."
    )

    chunks = chunk_text(
        text,
        chunk_size=75,
        chunk_overlap=50,
    )

    assert len(chunks) == 2

    assert chunks[0] == (
        "First sentence is here. "
        "Second sentence contains several important words."
    )

    assert chunks[1] == (
        "Second sentence contains several important words. "
        "Third sentence is here."
    )

def test_chunk_text_prefers_sentence_boundaries():
    text = (
        "Machine learning learns from data. "
        "Supervised learning uses labeled examples. "
        "Reinforcement learning learns through rewards."
    )

    chunks = chunk_text(
        text,
        chunk_size=60,
        chunk_overlap=0,
    )

    assert chunks[0] == "Machine learning learns from data."
    assert chunks[1] == (
        "Supervised learning uses labeled examples."
    )


def test_chunk_text_keeps_long_sentence_intact():
    text = (
        "Machine learning is a method where systems learn "
        "patterns directly from data without explicit rules."
    )

    chunks = chunk_text(
        text,
        chunk_size=20,
        chunk_overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0] == text


def test_build_chunks_assigns_metadata():
    chunks = build_chunks(
        "This is a document containing some test text.",
        document_id="example_doc",
        chunk_size=20,
        chunk_overlap=5,
    )

    assert chunks
    assert chunks[0].document_id == "example_doc"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "example_doc_chunk_000"
    assert chunks[0].text == chunks[0].text.strip()