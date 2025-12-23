import pdfplumber
import docx
import httpx
import tempfile
import os

async def parse_document_from_url(url: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        ext = url.split("?")[0].split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

    try:
        if ext == "pdf":
            with pdfplumber.open(tmp_path) as pdf:
                return "\n".join(page.extract_text() or '' for page in pdf.pages)

        elif ext == "docx":
            doc = docx.Document(tmp_path)
            return "\n".join(p.text for p in doc.paragraphs)

        else:
            return "Unsupported file type"
    finally:
        os.remove(tmp_path)