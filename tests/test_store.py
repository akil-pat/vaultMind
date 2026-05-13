from pathlib import Path

from vaultmind.ingest.store import VaultStore


def _records():
    return [
        {
            "id": "a.md::0",
            "path": "a.md",
            "heading": "Intro",
            "chunk_index": 0,
            "text": "vector databases store embeddings",
            "vector": [1.0, 0.0, 0.0],
        },
        {
            "id": "a.md::1",
            "path": "a.md",
            "heading": "Details",
            "chunk_index": 1,
            "text": "lancedb is an embedded vector database",
            "vector": [0.9, 0.1, 0.0],
        },
        {
            "id": "b.md::0",
            "path": "b.md",
            "heading": "Unrelated",
            "chunk_index": 0,
            "text": "recipe for banana bread",
            "vector": [0.0, 0.0, 1.0],
        },
    ]


def test_empty_store_returns_nothing(tmp_path: Path):
    store = VaultStore(tmp_path / "index.lance")

    assert store.is_empty()
    assert store.search([1.0, 0.0, 0.0]) == []
    assert store.list_paths() == []
    assert store.get_note_chunks("a.md") == []


def test_search_ranks_by_similarity(tmp_path: Path):
    store = VaultStore(tmp_path / "index.lance")
    store.rebuild(_records())

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert not store.is_empty()
    assert len(results) == 2
    assert results[0]["path"] == "a.md"
    assert results[0]["heading"] == "Intro"
    # closest match should score higher than the second-closest
    assert results[0]["score"] > results[1]["score"]


def test_list_paths_is_sorted_and_deduped(tmp_path: Path):
    store = VaultStore(tmp_path / "index.lance")
    store.rebuild(_records())

    assert store.list_paths() == ["a.md", "b.md"]


def test_get_note_chunks_filters_and_orders_by_index(tmp_path: Path):
    store = VaultStore(tmp_path / "index.lance")
    store.rebuild(_records())

    chunks = store.get_note_chunks("a.md")

    assert [c["chunk_index"] for c in chunks] == [0, 1]
    assert all(c["path"] == "a.md" for c in chunks)


def test_rebuild_replaces_previous_index(tmp_path: Path):
    store = VaultStore(tmp_path / "index.lance")
    store.rebuild(_records())
    assert len(store.list_paths()) == 2

    store.rebuild([_records()[0]])

    assert store.list_paths() == ["a.md"]


def test_rebuild_with_no_records_clears_store(tmp_path: Path):
    store = VaultStore(tmp_path / "index.lance")
    store.rebuild(_records())

    store.rebuild([])

    assert store.is_empty()
