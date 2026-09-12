import io
import fitz
from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text from a PDF."""
    text_parts = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    except Exception:
        pass
    return "\n\n".join(text_parts).strip()


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def render_pdf_pages(pdf_bytes: bytes, max_pages: int = 30, scale: float = 0.65):
    """Render PDF pages as compact JPEGs suitable for low-token vision OCR.

    Each page is sent separately to the vision model. Lower resolution and JPEG
    compression reduce image input tokens and make on-demand Groq limits easier
    to respect while keeping handwriting reasonably readable.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    try:
        for index, page in enumerate(doc):
            if index >= max_pages:
                break
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
                colorspace=fitz.csRGB,
            )
            images.append(pix.tobytes("jpeg", jpg_quality=50))
    finally:
        doc.close()
    return images
