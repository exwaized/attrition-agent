"""
Unit tests for rag/build_rag.py.

NOT covered: main() itself — needs a real ChromaDB path, a loaded
sentence-transformers model, and real policy .txt files on disk,
none of which a unit test should depend on (same reasoning as
rag/retriever.py's actual retrieval calls). Covered: chunk_text,
the one pure function in the file — chunking logic is exactly the
kind of thing that's easy to get subtly wrong at the boundaries
(overlap math, very short/empty input, chunks just under the
30-char minimum).
"""
import pytest

from rag.build_rag import chunk_text


def test_chunk_text_splits_long_text_into_multiple_chunks():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1


def test_chunk_text_respects_chunk_size():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_text_overlap_means_consecutive_chunks_share_content():
    text = "0123456789" * 40  # 400 chars, easy to reason about by index
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    # chunk 1 starts at char 250 (300-50), so its first 50 chars should
    # match the last 50 chars of chunk 0
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_drops_chunks_at_or_under_30_chars():
    # Text just long enough to produce one full chunk plus a tiny tail
    text = "A" * 310
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert all(len(c) > 30 for c in chunks)


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_chunk_text_short_text_under_minimum_returns_no_chunks():
    assert chunk_text("too short") == []  # under the 30-char floor


def test_chunk_text_strips_null_bytes():
    text = "A" * 50 + "\x00" + "B" * 50
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert "\x00" not in chunks[0]


def test_chunk_text_overlap_equal_to_chunk_size_raises_instead_of_hanging():
    # Before the fix, this combination made start += (chunk_size - overlap)
    # add zero on every iteration — start never advanced and the loop
    # never terminated. Confirmed by actually triggering it under a
    # hard timeout before fixing. Now it should fail fast and loud.
    with pytest.raises(ValueError):
        chunk_text("A" * 500, chunk_size=100, overlap=100)


def test_chunk_text_overlap_greater_than_chunk_size_also_raises():
    with pytest.raises(ValueError):
        chunk_text("A" * 500, chunk_size=100, overlap=150)
