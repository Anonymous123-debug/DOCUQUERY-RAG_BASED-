# app/parser.py
import io

import docx
import httpx
import pdfplumber

SUPPORTED_TYPES = ("pdf", "docx", "txt")


class UnsupportedDocumentError(ValueError):
    pass


def file_extension(url: str) -> str:
    return url.split("?")[0].split("#")[0].rsplit(".", 1)[-1].lower()


def extract_text(content: bytes, ext: str) -> str:
    if ext == "pdf":
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if ext == "docx":
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == "txt":
        return content.decode("utf-8", errors="ignore")
    raise UnsupportedDocumentError(f"Unsupported file type '.{ext}' (supported: {', '.join(SUPPORTED_TYPES)})")


async def parse_document_from_url(url: str) -> str:
    ext = file_extension(url)
    if ext not in SUPPORTED_TYPES:
        raise UnsupportedDocumentError(f"Unsupported file type '.{ext}' (supported: {', '.join(SUPPORTED_TYPES)})")

    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        r = await client.get(url)
        r.raise_for_status()

    return extract_text(r.content, ext)
