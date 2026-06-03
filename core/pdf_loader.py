"""
PDF text extraction using pypdf with page-number metadata tracking.
"""

from typing import List, Dict, Optional
from pypdf import PdfReader


class PDFLoader:
    """Extract text from PDF files page by page, preserving page numbers."""

    @staticmethod
    def load_pdf(file_path: str) -> List[Dict]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of dicts: [{"page_number": 1, "text": "Page content..."}, ...]
            Page numbers are 1-indexed.

        Raises:
            ValueError: If the PDF is encrypted or has no extractable text.
            FileNotFoundError: If the file does not exist.
        """
        reader = PdfReader(file_path)

        # Check for encryption
        if reader.is_encrypted:
            raise ValueError(
                "This PDF is password-protected and cannot be processed. "
                "Please upload an unprotected PDF."
            )

        pages: List[Dict] = []
        total_pages = len(reader.pages)
        has_text = False

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Normalize whitespace
            text = " ".join(text.split())
            if text.strip():
                has_text = True
            pages.append({
                "page_number": i + 1,  # 1-indexed for user-facing display
                "text": text,
                "total_pages": total_pages,
            })

        if not has_text:
            raise ValueError(
                "No extractable text found in this PDF. "
                "It may be a scanned document (image-only). "
                "OCR processing is required for scanned PDFs."
            )

        return pages

    @staticmethod
    def get_pdf_info(file_path: str) -> Dict:
        """
        Get basic metadata about a PDF without loading all pages.

        Returns:
            Dict with keys: filename, total_pages, is_encrypted.
        """
        reader = PdfReader(file_path)
        return {
            "filename": file_path,
            "total_pages": len(reader.pages),
            "is_encrypted": reader.is_encrypted,
        }
