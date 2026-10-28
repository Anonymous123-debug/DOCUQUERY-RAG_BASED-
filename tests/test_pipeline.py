import os

os.environ["API_TOKEN"] = "test-token"

from fastapi.testclient import TestClient

from app import main
from app.embedding import chunk_text, tokenizer
from app.parser import UnsupportedDocumentError, extract_text, file_extension

client = TestClient(main.app)


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(tokenizer.encode(c)) <= 100 for c in chunks)
    # The tail of one chunk reappears at the head of the next
    assert chunks[0].split()[-1] in chunks[1]


def test_chunk_rejects_overlap_not_smaller_than_size():
    try:
        chunk_text("hello", chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_file_extension_ignores_query_string():
    assert file_extension("https://x.com/policy.PDF?sv=2023&sig=abc") == "pdf"


def test_extract_txt_and_reject_unknown():
    assert extract_text(b"hello", "txt") == "hello"
    try:
        extract_text(b"", "xlsx")
    except UnsupportedDocumentError:
        return
    raise AssertionError("expected UnsupportedDocumentError")


def test_rejects_bad_token():
    r = client.post("/hackrx/run", json={"documents": "a.pdf", "questions": ["q"]},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_run_query_end_to_end(monkeypatch):
    async def fake_parse(url):
        return "The grace period is thirty days."

    upserted = {}
    monkeypatch.setattr(main, "is_indexed", lambda ns: False)
    monkeypatch.setattr(main, "parse_document_from_url", fake_parse)
    monkeypatch.setattr(main, "upsert_chunks", lambda ns, chunks: upserted.setdefault(ns, chunks) and len(chunks))
    monkeypatch.setattr(main, "search", lambda ns, q: upserted[ns])
    monkeypatch.setattr(main, "answer_question", lambda q, ctx: f"{q} -> {ctx[0]}")

    r = client.post("/hackrx/run",
                    json={"documents": "https://x.com/policy.pdf", "questions": ["Q1", "Q2"]},
                    headers={"Authorization": "Bearer test-token"})

    assert r.status_code == 200
    assert r.json() == {"answers": [
        "Q1 -> The grace period is thirty days.",
        "Q2 -> The grace period is thirty days.",
    ]}
