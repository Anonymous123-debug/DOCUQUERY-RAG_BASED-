# app/embedding.py

import os
import openai
import tiktoken
import uuid
import pinecone
from typing import List
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Optional: You can change this to another model
EMBED_MODEL = "text-embedding-3-small"
CHUNK_TOKEN_LIMIT = 300

tokenizer = tiktoken.encoding_for_model("gpt-4")

def chunk_text(text: str, chunk_size=CHUNK_TOKEN_LIMIT) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        token_len = len(tokenizer.encode(" ".join(current_chunk)))
        if token_len > chunk_size:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_embeddings(chunks: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        input=chunks,
        model=EMBED_MODEL
    )
    return [e.embedding for e in response.data]

def upsert_to_pinecone(index_name: str, chunks: List[str]):
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone.Index(index_name)

    vectors = []
    embeddings = get_embeddings(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    return len(vectors)
