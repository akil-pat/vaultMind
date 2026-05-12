from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vaultmind.ingest.loader import Document

_HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    path: Path
    heading: str
    text: str
    chunk_index: int

    @property
    def chunk_id(self) -> str:
        return f"{self.path}::{self.chunk_index}"


def chunk_document(doc: Document, chunk_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    """Split a document into overlapping, heading-aware chunks for embedding.

    Markdown files are first split on header boundaries so a chunk never
    straddles two unrelated sections; each section is then further split into
    fixed-size overlapping word windows if it's still too long. Non-markdown
    files are treated as a single untitled section.
    """
    if doc.path.suffix.lower() in {".md", ".markdown"}:
        sections = _split_markdown_sections(doc.text)
    else:
        sections = [(doc.path.stem, doc.text)]

    chunks: list[Chunk] = []
    index = 0
    for heading, body in sections:
        for window in _windows(body, chunk_words, overlap_words):
            if not window.strip():
                continue
            chunks.append(Chunk(path=doc.path, heading=heading, text=window, chunk_index=index))
            index += 1
    return chunks


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("(untitled)", text)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("(untitled)", preamble))

    for i, match in enumerate(matches):
        heading = match.group().lstrip("#").strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[body_start:body_end].strip()))
    return sections


def _windows(body: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = body.split()
    if len(words) <= chunk_words:
        return [body]

    step = max(chunk_words - overlap_words, 1)
    return [
        " ".join(words[start : start + chunk_words])
        for start in range(0, len(words), step)
        if start < len(words)
    ]
