# app/models.py
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    documents: str = Field(..., description="Public URL of a PDF, DOCX or TXT document")
    questions: list[str] = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answers: list[str]
