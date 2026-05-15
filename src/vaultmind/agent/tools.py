from __future__ import annotations

from pathlib import Path

from anthropic import beta_tool

from vaultmind.ingest.embedder import Embedder
from vaultmind.ingest.store import VaultStore


def _to_relative(abs_path: str, vault_path: Path) -> str:
    try:
        return str(Path(abs_path).relative_to(vault_path))
    except ValueError:
        return abs_path


def _to_absolute(rel_or_abs_path: str, vault_path: Path) -> str:
    p = Path(rel_or_abs_path)
    if p.is_absolute():
        return str(p)
    return str((vault_path / p).resolve())


def build_tools(store: VaultStore, embedder: Embedder, vault_path: Path) -> list:
    """Construct the three tools the agent uses to work with the vault.

    Closures over `store`/`embedder`/`vault_path` keep the @beta_tool functions
    free of any global state, so a fresh Agent (fresh vault, fresh model) never
    accidentally shares a tool with another instance.
    """

    @beta_tool
    def search_notes(query: str, top_k: int = 6) -> str:
        """Search the user's notes for content relevant to a query.

        Use this whenever the user's question might be answered by something
        they previously wrote down — do not rely on prior knowledge for
        anything that sounds personal, project-specific, or note-worthy.

        Args:
            query: What to search for, phrased as a natural-language
                description of the information needed, e.g. "docker
                deployment steps" or "ideas about RAG agents".
            top_k: Maximum number of matching note excerpts to return.
        """
        vector = embedder.embed_one(query)
        results = store.search(vector, top_k=top_k)
        if not results:
            return "No notes are indexed yet, or nothing matched this query."
        blocks = [
            f"[{_to_relative(r['path'], vault_path)} > {r['heading']}] (relevance={r['score']:.2f})\n{r['text']}"
            for r in results
        ]
        return "\n\n---\n\n".join(blocks)

    @beta_tool
    def read_note(path: str) -> str:
        """Read the full content of one note, reassembled from all its chunks.

        Use this when a search result excerpt isn't enough context to answer
        confidently and you need the whole note.

        Args:
            path: The note's path exactly as shown by search_notes or
                list_notes (a path relative to the vault root).
        """
        resolved = _to_absolute(path, vault_path)
        chunks = store.get_note_chunks(resolved)
        if not chunks:
            return f"No note found at path: {path}"
        return "\n\n".join(c["text"] for c in chunks)

    @beta_tool
    def list_notes() -> str:
        """List the paths of every note currently indexed in the vault.

        Use this to see what's available before searching, or when the user
        asks a broad question like "what have I written about?".
        """
        paths = store.list_paths()
        if not paths:
            return "The vault is empty — no notes have been ingested yet."
        return "\n".join(_to_relative(p, vault_path) for p in paths)

    return [search_notes, read_note, list_notes]
