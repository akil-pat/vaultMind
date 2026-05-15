from pathlib import Path

import pytest

from vaultmind.agent.tools import build_tools
from vaultmind.ingest.chunker import chunk_document
from vaultmind.ingest.embedder import Embedder
from vaultmind.ingest.loader import Document
from vaultmind.ingest.store import VaultStore


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


@pytest.fixture
def vault(tmp_path: Path, embedder: Embedder):
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "recipes.md").write_text(
        "# Banana Bread\n\nMix flour, sugar, and mashed banana. Bake at 350F.\n"
    )
    (vault_path / "work.md").write_text(
        "# RAG Agent Plan\n\nUse LanceDB for the vector store and Claude for the agent loop.\n"
    )

    store = VaultStore(vault_path / ".vaultmind" / "index.lance")
    records = []
    for doc_path in [vault_path / "recipes.md", vault_path / "work.md"]:
        doc = Document(path=doc_path, text=doc_path.read_text())
        for chunk in chunk_document(doc):
            vector = embedder.embed_one(chunk.text)
            records.append(
                {
                    "id": chunk.chunk_id,
                    "path": str(chunk.path),
                    "heading": chunk.heading,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "vector": vector,
                }
            )
    store.rebuild(records)
    return vault_path, store


def test_search_notes_finds_relevant_note(vault, embedder):
    vault_path, store = vault
    tools = build_tools(store, embedder, vault_path)
    search_notes = next(t for t in tools if t.name == "search_notes")

    result = search_notes(query="what vector database should I use for my agent?")

    assert "work.md" in result
    assert "LanceDB" in result


def test_search_notes_reports_empty_store(tmp_path, embedder):
    vault_path = tmp_path / "empty_vault"
    vault_path.mkdir()
    store = VaultStore(vault_path / ".vaultmind" / "index.lance")
    tools = build_tools(store, embedder, vault_path)
    search_notes = next(t for t in tools if t.name == "search_notes")

    result = search_notes(query="anything")

    assert "no notes" in result.lower()


def test_list_notes_returns_relative_paths(vault, embedder):
    vault_path, store = vault
    tools = build_tools(store, embedder, vault_path)
    list_notes = next(t for t in tools if t.name == "list_notes")

    result = list_notes()

    assert "recipes.md" in result
    assert "work.md" in result
    assert str(vault_path) not in result


def test_read_note_reconstructs_full_text_from_relative_path(vault, embedder):
    vault_path, store = vault
    tools = build_tools(store, embedder, vault_path)
    read_note = next(t for t in tools if t.name == "read_note")

    result = read_note(path="recipes.md")

    assert "banana" in result.lower()
    assert "350F" in result


def test_read_note_missing_path_returns_friendly_message(vault, embedder):
    vault_path, store = vault
    tools = build_tools(store, embedder, vault_path)
    read_note = next(t for t in tools if t.name == "read_note")

    result = read_note(path="does_not_exist.md")

    assert "no note found" in result.lower()
