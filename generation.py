"""STEP 4 - The G in RAG: give the retrieved chunks + question to the local LLM."""
from openai import OpenAI

from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)   # points to LM Studio, not OpenAI

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using ONLY the numbered context passages. "
    "If the answer is not in the context, say: \"I don't know based on the provided documents.\" "
    "Cite the passages you used like [1] or [2]. Do not use outside knowledge."
)


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def generate_answer(question: str, hits: list[dict]) -> str:
    user_prompt = f"Context:\n{build_context(hits)}\n\nQuestion: {question}\n\nAnswer:"
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,     # 0 = most deterministic, best for factual RAG
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()
