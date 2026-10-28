# app/llm.py
from typing import List

from app import config
from app.embedding import openai_client

SYSTEM_PROMPT = (
    "You answer questions about a document using only the excerpts provided. "
    "Answer in one or two concise sentences. If the excerpts do not contain the answer, "
    "reply exactly: The document does not provide this information."
)


def answer_question(question: str, context: List[str]) -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context))
    response = openai_client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{excerpts}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()
