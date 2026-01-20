from __future__ import annotations

from typing import Iterable, List, Tuple
from uuid import uuid4

from .models import DocumentChunk, DocumentMetadata


def simple_paragraph_chunker(
    text: str,
    metadata: DocumentMetadata,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[DocumentChunk]:
    """
    Simple whitespace / paragraph-based chunker with character count limits.
    Chunking strategy is deliberately straightforward but configurable.
    """
    paragraphs: List[str] = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[DocumentChunk] = []
    buffer: List[str] = []
    current_len = 0
    chunk_index = 0

    def flush_buffer(buf: List[str], idx: int) -> Tuple[DocumentChunk, str]:
        joined = "\n\n".join(buf).strip()
        chunk = DocumentChunk(
            chunk_id=str(uuid4()),
            text=joined,
            metadata=metadata,
            chunk_index=idx,
        )
        # For overlap, keep last `chunk_overlap` characters.
        overlap_text = joined[-chunk_overlap:] if chunk_overlap > 0 else ""
        return chunk, overlap_text

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_size and buffer:
            chunk, overlap_text = flush_buffer(buffer, chunk_index)
            chunks.append(chunk)
            chunk_index += 1
            buffer = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)

        buffer.append(para)
        current_len += para_len

    if buffer:
        chunk, _ = flush_buffer(buffer, chunk_index)
        chunks.append(chunk)

    return chunks


def chunk_documents(
    docs: Iterable[Tuple[str, str, DocumentMetadata]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[DocumentChunk]:
    """
    Chunk an iterable of (doc_id, text, metadata) tuples.
    """
    all_chunks: List[DocumentChunk] = []
    for _, text, metadata in docs:
        all_chunks.extend(
            simple_paragraph_chunker(
                text=text,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks

