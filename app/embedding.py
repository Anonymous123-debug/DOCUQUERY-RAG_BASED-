# app/embedding.py
import hashlib
from functools import lru_cache
from typing import List

import tiktoken
from openai import OpenAI
from pinecone import Pinecone

from app import config

EMBED_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 100

tokenizer = tiktoken.get_encoding("cl100k_base")


@lru_cache
def openai_client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


@lru_cache
def pinecone_index():
    return Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)


def chunk_text(text: str, chunk_size: int = config.CHUNK_TOKENS, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Split text into windows of `chunk_size` tokens that overlap by `overlap` tokens."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = tokenizer.encode(text)
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = tokenizer.decode(tokens[start:start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        response = openai_client().embeddings.create(input=texts[i:i + EMBED_BATCH_SIZE], model=config.EMBEDDING_MODEL)
        embeddings.extend(e.embedding for e in response.data)
    return embeddings


def namespace_for(document_url: str) -> str:
    """One Pinecone namespace per document, so each document is embedded only once."""
    return hashlib.sha256(document_url.encode()).hexdigest()[:32]


def is_indexed(namespace: str) -> bool:
    return namespace in pinecone_index().describe_index_stats().namespaces


def upsert_chunks(namespace: str, chunks: List[str]) -> int:
    index = pinecone_index()
    embeddings = get_embeddings(chunks)
    vectors = [
        {"id": f"{namespace}-{i}", "values": embedding, "metadata": {"text": chunk}}
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        index.upsert(vectors=vectors[i:i + UPSERT_BATCH_SIZE], namespace=namespace)
    return len(vectors)


def search(namespace: str, question: str, top_k: int = config.TOP_K) -> List[str]:
    [embedding] = get_embeddings([question])
    result = pinecone_index().query(vector=embedding, top_k=top_k, namespace=namespace, include_metadata=True)
    return [match.metadata["text"] for match in result.matches]
