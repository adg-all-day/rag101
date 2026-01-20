from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, Tuple

from .models import DocumentMetadata


def load_text_files(
    root: Path,
) -> Iterable[Tuple[str, str, DocumentMetadata]]:
    """
    Load .txt and .md files from a directory tree.
    This is a minimal, robust loader you can extend.
    """
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(
            doc_id=doc_id,
            doc_title=path.stem,
            source=str(path.relative_to(root)),
        )
        yield doc_id, text, metadata

