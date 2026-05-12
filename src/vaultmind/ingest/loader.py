from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}
SKIP_DIRS = {".git", ".vaultmind", "node_modules", ".obsidian"}


@dataclass
class Document:
    path: Path
    text: str


def load_vault(vault_path: Path) -> Iterator[Document]:
    """Walk a directory and yield every readable note as a Document.

    Supports Markdown, plain text, and PDF. Files that fail to parse (corrupt
    PDF, binary garbage) are skipped rather than aborting the whole ingest run.
    """
    for path in sorted(vault_path.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif suffix in PDF_SUFFIXES:
                text = _read_pdf(path)
            else:
                continue
        except OSError:
            continue
        if text.strip():
            yield Document(path=path, text=text)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
