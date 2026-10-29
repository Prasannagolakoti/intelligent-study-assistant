"""
Unit Tests for DocumentProcessor
Tests validation, extraction (PDF, TXT), cleaning, metrics, and chunking.
"""

import os
import pytest
from modules.document_processor import DocumentProcessor, DocumentProcessingError


@pytest.fixture
def processor():
    return DocumentProcessor(max_file_size_bytes=5 * 1024 * 1024)  # 5 MB test limit


def test_file_validation_valid(processor):
    """Test valid PDF and TXT file extensions."""
    valid, err = processor.validate_file("notes.pdf", 1024)
    assert valid is True
    assert err is None

    valid, err = processor.validate_file("lecture.txt", 2048)
    assert valid is True
    assert err is None


def test_file_validation_invalid_extension(processor):
    """Test rejection of unsupported formats."""
    valid, err = processor.validate_file("virus.exe", 1024)
    assert valid is False
    assert "Unsupported file format" in err

    valid, err = processor.validate_file("image.png", 5000)
    assert valid is False
    assert "Unsupported file format" in err


def test_file_validation_empty_and_size_limits(processor):
    """Test rejection of empty files and oversized files."""
    valid, err = processor.validate_file("empty.pdf", 0)
    assert valid is False
    assert "empty" in err.lower()

    # Oversized
    valid, err = processor.validate_file("huge.pdf", 10 * 1024 * 1024)
    assert valid is False
    assert "exceeds" in err.lower()


def test_extract_from_txt_utf8(processor):
    """Test text extraction from UTF-8 encoded text bytes."""
    sample_text = "Operating systems manage hardware resources. Semaphores prevent race conditions."
    raw_bytes = sample_text.encode("utf-8")
    extracted = processor.extract_from_txt(raw_bytes)
    assert extracted == sample_text


def test_extract_from_txt_latin1(processor):
    """Test fallback encoding with Latin-1 bytes."""
    sample_text = "Café synchronization and résumé parsing in OS theory."
    raw_bytes = sample_text.encode("latin-1")
    extracted = processor.extract_from_txt(raw_bytes)
    assert "Café" in extracted or "Caf" in extracted


def test_extract_from_txt_empty(processor):
    """Test error raised on empty text file bytes."""
    with pytest.raises(DocumentProcessingError):
        processor.extract_from_txt(b"")


def test_extract_from_real_pdf(processor):
    """Test extraction from actual generated PDF file."""
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "operating_systems.pdf")
    assert os.path.exists(pdf_path), "Sample PDF should exist"

    with open(pdf_path, "rb") as f:
        content = f.read()

    text, page_count = processor.extract_from_pdf(content)
    assert page_count >= 1
    assert "Process Synchronization" in text or "Operating Systems" in text
    assert len(text) > 500


def test_extract_from_corrupted_pdf(processor):
    """Test error handling for corrupted PDF data."""
    bogus_bytes = b"%PDF-1.4\nCorrupted binary noise that is not a valid PDF object..."
    with pytest.raises(DocumentProcessingError):
        processor.extract_from_pdf(bogus_bytes)


def test_text_cleaning(processor):
    """Test whitespace normalization and junk character removal."""
    dirty_text = "Line 1   with   spaces.\r\n\r\n\r\n\r\nLine 2 with \t tabs and \x00 null bytes."
    cleaned = processor.clean_text(dirty_text)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Line 1 with spaces." in cleaned
    assert "Line 2 with tabs and null bytes." in cleaned


def test_document_metrics(processor):
    """Test document statistics calculation."""
    sample = "Deadlocks occur when processes wait indefinitely. A semaphore is an integer variable. Progress is essential."
    metrics = processor.get_document_metrics(sample, page_count=1)
    assert metrics["word_count"] > 10
    assert metrics["sentence_count"] == 3
    assert metrics["estimated_reading_minutes"] >= 1
    assert metrics["page_count"] == 1


def test_long_document_chunking(processor):
    """Test chunking of long documents into manageable chunks."""
    # Generate 500 words
    sentences = [f"This is test academic sentence number {i} discussing algorithm behavior." for i in range(1, 100)]
    long_text = "\n\n".join(sentences)

    chunks = processor.chunk_text(long_text, max_chunk_words=100, overlap_words=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "text" in chunk
        assert "word_count" in chunk
        assert chunk["word_count"] > 0
