from __future__ import annotations

import os
from typing import List, Tuple

from openai import OpenAI

from .config import settings
from .models import (
    AnswerRequest,
    AnswerResponse,
    QueryRequest,
    SourceCitation,
)
from .query_pipeline import run_query


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv(settings.llm.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"LLM API key not configured. "
            f"Set environment variable {settings.llm.api_key_env}."
        )

    _client = OpenAI(api_key=api_key)
    return _client


def _build_context_and_citations(
    query: str,
    top_k: int | None = None,
) -> Tuple[str, List[SourceCitation], QueryRequest]:
    retrieval_req = QueryRequest(query=query, top_k=top_k)
    retrieval_res = run_query(retrieval_req)

    lines: List[str] = ["Context:"]
    citations: List[SourceCitation] = []

    for idx, item in enumerate(retrieval_res.results, start=1):
        md = item.chunk.metadata
        label = f"Source {idx}"

        doc_label = md.doc_title or md.source or md.doc_id
        header = f"[{label} - Document: \"{doc_label}\""
        if md.page_number is not None:
            header += f", Page {md.page_number}"
        header += "]"

        lines.append(header)
        lines.append(item.chunk.text)
        lines.append("")

        citations.append(
            SourceCitation(
                source_id=idx,
                doc_title=md.doc_title,
                source=md.source,
                page_number=md.page_number,
                section=md.section,
                score=item.score,
            )
        )

    context_str = "\n".join(lines)
    return context_str, citations, retrieval_req


def answer_question(req: AnswerRequest) -> AnswerResponse:
    context, citations, retrieval_req = _build_context_and_citations(req.query, top_k=req.top_k)

    system_prompt = (
        "You are a helpful assistant that answers questions based on the "
        "provided context. Always cite your sources using [Source N] notation. "
        "If the answer is not in the context, clearly say you don't know."
    )

    user_prompt = (
        f"{context}\n\n"
        f"Question: {req.query}\n\n"
        "Instructions:\n"
        "- Answer based solely on the provided context.\n"
        "- Cite sources using [Source N] format.\n"
        "- If information is not in the context, state "
        "\"I don't have information about that\".\n"
        "- Be concise and accurate.\n\n"
        "Answer:"
    )

    client = _get_client()

    completion = client.chat.completions.create(
        model=settings.llm.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )

    answer_text = completion.choices[0].message.content or ""

    retrieval_results = None
    if req.include_chunks:
        # Reuse the retrieval request to get structured chunk results.
        retrieval_results = run_query(retrieval_req).results

    return AnswerResponse(
        query=req.query,
        answer=answer_text,
        sources=citations,
        retrieval_results=retrieval_results,
    )
