"""
rag/loader.py
-------------
Responsible for turning raw input (uploaded PDFs or pasted text) into clean,
page-aware text. Keeping extraction isolated here means the rest of the
pipeline never has to know whether content came from a PDF or plain text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """A single page (or pseudo-page, for pasted text) of extracted content."""

    page_number: int
    text: str
    source: str  # filename or "Pasted Text"


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or parsed."""


class PDFLoader:
    """Extracts text from PDF files, with an OCR fallback for scanned/image PDFs."""

    def __init__(self, ocr_enabled: bool = True, tesseract_cmd: Optional[str] = None):
        self.ocr_enabled = ocr_enabled
        self._ocr_available = False

        if ocr_enabled:
            try:
                import pytesseract  # noqa: F401
                from PIL import Image  # noqa: F401

                if tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                self._ocr_available = True
            except ImportError:
                logger.warning("OCR dependencies not available; image-only PDFs will be skipped.")
                self._ocr_available = False

    def load(self, file_path: str, source_name: Optional[str] = None) -> List[PageContent]:
        """
        Extract text from a PDF, page by page.

        Falls back to OCR for pages with no extractable text (common in
        scanned documents), if OCR dependencies are available.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        name = source_name or path.name
        pages: List[PageContent] = []

        try:
            doc = fitz.open(file_path)
        except Exception as exc:  # PyMuPDF raises generic exceptions on corrupt files
            raise DocumentLoadError(f"Could not open '{name}'. It may be corrupted or password-protected.") from exc

        if doc.page_count == 0:
            doc.close()
            raise DocumentLoadError(f"'{name}' contains no pages.")

        try:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()

                if not text and self._ocr_available:
                    text = self._ocr_page(page)
                    if text:
                        logger.info("Page %d of '%s' recovered via OCR.", i, name)

                cleaned = clean_text(text)
                if cleaned:
                    pages.append(PageContent(page_number=i, text=cleaned, source=name))
        finally:
            doc.close()

        if not pages:
            raise DocumentLoadError(
                f"No extractable text found in '{name}'. It may be a blank or fully image-based "
                f"PDF with OCR unavailable."
            )

        return pages

    def _ocr_page(self, page: "fitz.Page") -> str:
        """Render a page to an image and run OCR on it."""
        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img).strip()
        except Exception as exc:
            logger.warning("OCR failed for a page: %s", exc)
            return ""


def clean_text(text: str) -> str:
    """Normalize whitespace and strip common PDF extraction artifacts."""
    if not text:
        return ""
    # Collapse repeated whitespace, but preserve paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove stray null/control characters sometimes left by PDF extraction.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def load_pasted_text(text: str, source_name: str = "Pasted Text") -> List[PageContent]:
    """Wrap pasted plain text in the same PageContent structure as PDF pages."""
    cleaned = clean_text(text)
    if not cleaned:
        raise DocumentLoadError("Pasted text is empty after cleaning.")
    return [PageContent(page_number=1, text=cleaned, source=source_name)]
