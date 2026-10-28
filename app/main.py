# app/main.py
import asyncio
import logging
import secrets

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import config
from app.embedding import chunk_text, is_indexed, namespace_for, search, upsert_chunks
from app.llm import answer_question
from app.models import QueryRequest, QueryResponse
from app.parser import UnsupportedDocumentError, parse_document_from_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocuQuery", description="Question answering over documents with retrieval-augmented generation")
security = HTTPBearer()


def verify_token(auth: HTTPAuthorizationCredentials = Depends(security)) -> None:
    if not config.API_TOKEN or not secrets.compare_digest(auth.credentials, config.API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")


async def ingest(document_url: str) -> str:
    namespace = namespace_for(document_url)
    if await asyncio.to_thread(is_indexed, namespace):
        logger.info("Document already indexed (namespace=%s)", namespace)
        return namespace

    try:
        text = await parse_document_from_url(document_url)
    except UnsupportedDocumentError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Could not download document: {e}")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from the document")

    count = await asyncio.to_thread(upsert_chunks, namespace, chunks)
    logger.info("Indexed %d chunks (namespace=%s)", count, namespace)
    return namespace


async def answer(namespace: str, question: str) -> str:
    context = await asyncio.to_thread(search, namespace, question)
    return await asyncio.to_thread(answer_question, question, context)


@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(request: QueryRequest) -> QueryResponse:
    namespace = await ingest(request.documents)
    answers = await asyncio.gather(*(answer(namespace, q) for q in request.questions))
    return QueryResponse(answers=list(answers))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
