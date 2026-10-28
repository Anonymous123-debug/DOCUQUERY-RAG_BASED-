# DocuQuery

A retrieval-augmented generation (RAG) API that answers natural-language questions about long documents. Give it a link to a PDF, DOCX or TXT file and a list of questions, and it returns one short answer per question, drawn only from the document's text.

Built with FastAPI, OpenAI embeddings and chat models, and the Pinecone vector database.

## How it works

```
document URL ─► download ─► extract text ─► token chunking ─► embed ─► Pinecone (namespace per document)
                              (pdfplumber /   (300 tokens,      (text-embedding-
                               python-docx)    50 overlap)        3-small, batched)

question ─► embed ─► top-k similarity search ─► grounded prompt ─► LLM answer
```

1. **Ingestion.** The document is downloaded asynchronously and its text is extracted. The text is split into overlapping 300-token windows with `tiktoken`, then embedded and upserted to Pinecone in batches of 100.
2. **Caching.** Each document gets its own Pinecone namespace, named from a SHA-256 hash of its URL. When the same document is queried again, the pipeline skips download, parsing and embedding.
3. **Retrieval and answering.** Each question is embedded and matched against the document's top-5 chunks. The LLM answers at temperature 0 using only those excerpts. If the excerpts don't contain the answer, it says the document does not provide this information instead of guessing.
4. **Concurrency.** All questions in a request are answered in parallel with `asyncio.gather`.

## API

`POST /hackrx/run` (requires `Authorization: Bearer <API_TOKEN>`)

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "Does the policy cover maternity expenses?"
  ]
}
```

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment.",
    "Yes, maternity expenses are covered after 24 months of continuous coverage."
  ]
}
```

| Status | Meaning |
| --- | --- |
| 401 | Missing or wrong bearer token |
| 400 | Document could not be downloaded |
| 415 | Unsupported file type |
| 422 | No text could be extracted, or the request body is invalid |

`GET /health` returns `{"status": "ok"}`.

## Setup

You need Python 3.11 or later, an OpenAI API key, and a Pinecone index with **1536 dimensions** and the **cosine** metric (this matches `text-embedding-3-small`).

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # then fill in your keys
uvicorn app.main:app --reload
```

The interactive docs are served at http://localhost:8000/docs.

Configuration is read from `.env`. See [.env.example](.env.example) for every option, including the models, chunk size, overlap and top-k.

## Tests

```bash
pytest
```

The tests cover chunking, file-type handling, auth, and the full request flow with the external services mocked.

## Project structure

```
app/
  main.py        FastAPI app: auth, ingestion and the query endpoint
  parser.py      Document download and text extraction
  embedding.py   Chunking, embeddings, Pinecone upsert and search
  llm.py         Grounded answer generation
  models.py      Request and response schemas
  config.py      Environment-based settings
tests/
```
