# app/main.py
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from app.parser import parse_document_from_url
from app.embedding import chunk_text, upsert_to_pinecone

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def run_query(request: QueryRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    text = await parse_document_from_url(request.documents)
    return {"message": "Document parsed", "length": len(text)}


text = await parse_document_from_url(request.documents)
chunks = chunk_text(text)
count = upsert_to_pinecone(index_name="your-index-name", chunks=chunks)
