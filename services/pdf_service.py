import io
from typing import List, Dict

import fitz
from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all selectable text from a PDF locally. No API call."""
    parts = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    except Exception:
        pass
    return "\n\n".join(parts).strip()


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return len(doc)
    except Exception:
        return 0


def extract_pages_with_text(pdf_bytes: bytes, min_chars: int = 25) -> List[Dict]:
    """Return per-page selectable text so mixed PDFs only OCR the pages that need it."""
    pages = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append({
                "page": i,
                "text": text,
                "needs_ocr": len(text) < min_chars,
            })
    except Exception:
        return []
    return pages


def render_pdf_pages(
    pdf_bytes: bytes,
    max_pages: int = 30,
    scale: float = 0.60,
    page_numbers: List[int] | None = None,
):
    """Render selected PDF pages as compact JPEGs.

    page_numbers uses 1-based page numbers. If omitted, all pages up to max_pages
    are rendered. Compact images reduce vision input tokens and upload time.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    wanted = set(page_numbers) if page_numbers else None
    try:
        for index, page in enumerate(doc):
            page_number = index + 1
            if page_number > max_pages:
                break
            if wanted is not None and page_number not in wanted:
                continue
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
                colorspace=fitz.csRGB,
            )
            images.append({
                "page": page_number,
                "bytes": pix.tobytes("jpeg", jpg_quality=48),
            })
    finally:
        doc.close()
    return images
