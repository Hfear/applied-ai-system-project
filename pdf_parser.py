"""
PDF text extraction utility using pypdf.
Gracefully returns an empty string on any failure.
"""

from __future__ import annotations


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF given its raw bytes.

    Returns concatenated page text, or an empty string if extraction fails.
    """
    try:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception:
        return ""
