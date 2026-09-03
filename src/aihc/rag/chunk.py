"""Chunking with stable IDs.

A chunk ID is derived from the file name and the chunk's own content hash, so the same
document always yields the same IDs. Citations in cached answers stay valid until the
underlying text changes, at which point the fingerprint changes and the cache misses anyway.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..config import DOCS_DIR


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def _split(text: str, max_chars: int = 600) -> list[str]:
    """Paragraph-aware split. Keeps headings attached to the paragraph that follows."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if p.startswith("#") and not buf:
            buf = p
            continue
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def load_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for piece in _split(path.read_text(encoding="utf-8")):
            h = hashlib.sha1((path.name + piece).encode()).hexdigest()[:8]
            chunks.append(Chunk(id=f"{path.stem}#{h}", source=path.name, text=piece))
    return chunks
