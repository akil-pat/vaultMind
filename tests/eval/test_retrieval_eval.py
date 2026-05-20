"""Retrieval quality eval — the piece most portfolio RAG projects skip.

This isn't a unit test of a single function; it's a small, checked-in QA set
that measures whether the *whole ingestion+retrieval pipeline* actually
surfaces the right note for a realistic question. If someone changes the
chunking strategy, the embedding model, or `top_k` and retrieval quality
regresses, this is what catches it — a passing unit test on chunk_document()
in isolation wouldn't.
"""

import json
from pathlib import Path

import pytest

from vaultmind.ingest.chunker import chunk_document
from vaultmind.ingest.embedder import Embedder
from vaultmind.ingest.loader import load_vault
from vaultmind.ingest.store import VaultStore

SAMPLE_VAULT = Path(__file__).parents[2] / "examples" / "sample_vault"
QA_SET = json.loads((Path(__file__).parent / "qa_set.json").read_text())
EVAL_TOP_K = 3
MIN_HIT_RATE = 0.85  # embeddings aren't perfect; a handful of near-misses is expected drift, not a bug


@pytest.fixture(scope="module")
def indexed_sample_vault(tmp_path_factory):
    """Index the checked-in sample vault once per test session, into a
    throwaway index (not the repo's examples/sample_vault/.vaultmind) so
    running the eval never leaves generated files for git to see.
    """
    embedder = Embedder()
    store = VaultStore(tmp_path_factory.mktemp("eval_index") / "index.lance")

    records = []
    for doc in load_vault(SAMPLE_VAULT):
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
    return store, embedder


def test_sample_vault_has_expected_notes():
    # Guards against someone renaming/removing a sample note without
    # updating qa_set.json — a silent eval-set/fixture mismatch otherwise.
    docs = {doc.path.name for doc in load_vault(SAMPLE_VAULT)}
    expected = {item["expected_path"] for item in QA_SET}
    assert expected <= docs


@pytest.mark.parametrize("case", QA_SET, ids=[c["question"] for c in QA_SET])
def test_retrieval_surfaces_the_right_note(indexed_sample_vault, case):
    store, embedder = indexed_sample_vault
    vector = embedder.embed_one(case["question"])
    results = store.search(vector, top_k=EVAL_TOP_K)

    retrieved_paths = [Path(r["path"]).name for r in results]
    assert case["expected_path"] in retrieved_paths, (
        f"expected {case['expected_path']!r} in top-{EVAL_TOP_K} for "
        f"{case['question']!r}, got {retrieved_paths}"
    )

    hit = next(r for r in results if Path(r["path"]).name == case["expected_path"])
    assert case["expected_keyword"].lower() in hit["text"].lower()


def test_overall_hit_rate_meets_threshold(indexed_sample_vault):
    store, embedder = indexed_sample_vault
    hits = 0
    for case in QA_SET:
        vector = embedder.embed_one(case["question"])
        results = store.search(vector, top_k=EVAL_TOP_K)
        retrieved_paths = {Path(r["path"]).name for r in results}
        if case["expected_path"] in retrieved_paths:
            hits += 1

    hit_rate = hits / len(QA_SET)
    print(f"\nretrieval hit@{EVAL_TOP_K}: {hits}/{len(QA_SET)} ({hit_rate:.0%})")
    assert hit_rate >= MIN_HIT_RATE
