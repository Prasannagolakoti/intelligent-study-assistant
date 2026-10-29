"""
Document Processor Module
Handles document validation, text extraction (PDF, TXT), text cleaning,
statistical analysis, and intelligent chunking for long study materials.
"""

import io
import re
from typing import Dict, List, Optional, Tuple, Any

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB default limit


class DocumentProcessingError(Exception):
    """Custom exception raised when document processing fails."""
    pass


class DocumentProcessor:
    """
    Robust processor for student study materials.
    Validates file integrity, extracts text, cleans formatting,
    computes reading metrics, and handles large document chunking.
    """

    def __init__(self, max_file_size_bytes: int = MAX_FILE_SIZE_BYTES):
        self.max_file_size_bytes = max_file_size_bytes

    def validate_file(self, filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate file extension and file size.
        Returns (is_valid, error_message).
        """
        if not filename:
            return False, "Filename cannot be empty."

        lower_name = filename.lower()
        if not (lower_name.endswith('.pdf') or lower_name.endswith('.txt')):
            return False, f"Unsupported file format '{filename}'. Please upload a .pdf or .txt file."

        if file_size <= 0:
            return False, "The uploaded file is empty (0 bytes)."

        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes // (1024 * 1024)
            return False, f"File exceeds maximum allowed size of {max_mb} MB."

        return True, None

    def extract_from_txt(self, file_content: bytes) -> str:
        """
        Extract text from TXT bytes with multi-encoding fallback.
        """
        if not file_content:
            raise DocumentProcessingError("Text file is empty.")

        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        for enc in encodings:
            try:
                text = file_content.decode(enc)
                return text
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback with replacement
        return file_content.decode('utf-8', errors='replace')

    def extract_from_pdf(self, file_content: bytes) -> Tuple[str, int]:
        """
        Extract text from PDF bytes using pypdf.
        Returns (extracted_text, page_count).
        Handles corrupted files, password-protected PDFs, and scanned PDFs.
        """
        if not file_content:
            raise DocumentProcessingError("Uploaded PDF file is empty (0 bytes).")

        if not PYPDF_AVAILABLE:
            raise DocumentProcessingError(
                "pypdf library is not installed in the environment. "
                "Please run `pip install pypdf` to enable PDF support."
            )

        try:
            stream = io.BytesIO(file_content)
            reader = pypdf.PdfReader(stream)
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read PDF file. The file may be corrupted or not a valid PDF: {str(e)}"
            )

        if reader.is_encrypted:
            try:
                # Attempt empty password decryption
                decrypted = reader.decrypt("")
                if not decrypted:
                    raise DocumentProcessingError("PDF is password-protected and cannot be read without credentials.")
            except Exception:
                raise DocumentProcessingError("PDF is password-protected. Please upload an unlocked PDF.")

        page_count = len(reader.pages)
        if page_count == 0:
            raise DocumentProcessingError("PDF contains 0 pages.")

        extracted_pages = []
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text)
            except Exception as page_err:
                extracted_pages.append(f"[Warning: Could not extract page {idx+1}: {str(page_err)}]")

        full_text = "\n\n".join(extracted_pages).strip()

        # Scanned PDF detection: if total non-whitespace characters are negligible
        clean_chars = re.sub(r'\s+', '', full_text)
        if len(clean_chars) < 25:
            raise DocumentProcessingError(
                "No extractable text found in this PDF. It appears to be a scanned image or empty document. "
                "Please upload a searchable text-based PDF or paste your study notes directly."
            )

        return full_text, page_count

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize raw extracted text:
        - Replaces non-standard whitespace/tabs with spaces
        - Collapses 3+ consecutive newlines to 2 (preserves paragraph separation)
        - Removes non-printable characters while preserving standard accents & punctuation
        - Strips surrounding whitespace
        """
        if not text:
            return ""

        # Normalize carriage returns
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Replace non-breaking spaces and special unicode spaces
        text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)
        text = text.replace('\xa0', ' ')

        # Remove form feed and other control characters (keep \n and \t)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Collapse excessive spaces on individual lines
        text = re.sub(r'[ \t]+', ' ', text)

        # Collapse more than 2 consecutive newlines into 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def get_document_metrics(self, text: str, page_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute descriptive metrics for the processed document:
        - Word count
        - Character count
        - Sentence count
        - Paragraph count
        - Estimated reading time (average 200 words per minute for academic material)
        - Page count (if provided)
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return {
                "word_count": 0,
                "char_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "estimated_reading_minutes": 0,
                "page_count": page_count or 0,
            }

        words = re.findall(r'\b\w+\b', cleaned)
        word_count = len(words)
        char_count = len(cleaned)

        # Approximate sentence split on punctuation followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # Paragraph split on double newlines
        paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)

        # Academic reading rate: ~180-200 wpm
        reading_minutes = max(1, round(word_count / 190, 1)) if word_count > 0 else 0

        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "estimated_reading_minutes": reading_minutes,
            "page_count": page_count or 1,
        }

    def chunk_text(
        self,
        text: str,
        max_chunk_words: int = 600,
        overlap_words: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Split long documents into contextual chunks suitable for LLM context windows
        and extractive algorithms.
        Preserves paragraph and sentence boundaries wherever possible.
        Each chunk includes metadata (index, word count, character range).
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Split into paragraphs first
        raw_paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [cleaned]

        chunks: List[Dict[str, Any]] = []
        current_chunk_paragraphs: List[str] = []
        current_word_count = 0

        for para in raw_paragraphs:
            para_words = len(re.findall(r'\b\w+\b', para))

            # If a single paragraph is larger than max_chunk_words, break it into sentences
            if para_words > max_chunk_words:
                # Flush existing chunk if any
                if current_chunk_paragraphs:
                    chunk_str = "\n\n".join(current_chunk_paragraphs)
                    chunks.append({
                        "index": len(chunks) + 1,
                        "text": chunk_str,
                        "word_count": current_word_count
                    })
                    current_chunk_paragraphs = []
                    current_word_count = 0

                # Break the oversized paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub_chunk: List[str] = []
                sub_word_count = 0
                for sent in sentences:
                    sent_words = len(re.findall(r'\b\w+\b', sent))
                    if sub_word_count + sent_words > max_chunk_words and sub_chunk:
                        chunks.append({
                            "index": len(chunks) + 1,
                            "text": " ".join(sub_chunk),
                            "word_count": sub_word_count
                        })
                        # keep last sentence for overlap
                        sub_chunk = [sent]
                        sub_word_count = sent_words
                    else:
                        sub_chunk.append(sent)
                        sub_word_count += sent_words

                if sub_chunk:
                    chunks.append({
                        "index": len(chunks) + 1,
                        "text": " ".join(sub_chunk),
                        "word_count": sub_word_count
                    })
                continue

            if current_word_count + para_words > max_chunk_words and current_chunk_paragraphs:
                chunk_str = "\n\n".join(current_chunk_paragraphs)
                chunks.append({
                    "index": len(chunks) + 1,
                    "text": chunk_str,
                    "word_count": current_word_count
                })

                # Retain overlap from previous paragraph if overlap requested
                if overlap_words > 0 and current_chunk_paragraphs:
                    overlap_para = current_chunk_paragraphs[-1]
                    current_chunk_paragraphs = [overlap_para, para]
                    current_word_count = len(re.findall(r'\b\w+\b', overlap_para)) + para_words
                else:
                    current_chunk_paragraphs = [para]
                    current_word_count = para_words
            else:
                current_chunk_paragraphs.append(para)
                current_word_count += para_words

        if current_chunk_paragraphs:
            chunk_str = "\n\n".join(current_chunk_paragraphs)
            chunks.append({
                "index": len(chunks) + 1,
                "text": chunk_str,
                "word_count": current_word_count
            })

        return chunks
