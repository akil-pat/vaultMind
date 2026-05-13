from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import lancedb


class SearchResult(TypedDict):
    id: str
    path: str
    heading: str
    chunk_index: int
    text: str
    score: float


TABLE_NAME = "chunks"


class VaultStore:
    """Embedded vector store for note chunks, backed by LanceDB on disk.

    One store = one vault. `rebuild()` fully replaces the index, which keeps
    ingestion idempotent — re-running it after editing notes never leaves
    stale chunks behind.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))

    def rebuild(self, records: list[dict]) -> None:
        if not records:
            self.db.drop_table(TABLE_NAME, ignore_missing=True)
            return
        self.db.create_table(TABLE_NAME, data=records, mode="overwrite")

    def is_empty(self) -> bool:
        return TABLE_NAME not in self.db.list_tables().tables

    def search(self, vector: list[float], top_k: int = 6) -> list[SearchResult]:
        if self.is_empty():
            return []
        rows = self.db.open_table(TABLE_NAME).search(vector).limit(top_k).to_list()
        return [
            SearchResult(
                id=row["id"],
                path=row["path"],
                heading=row["heading"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                score=1.0 - row["_distance"],
            )
            for row in rows
        ]

    def get_note_chunks(self, path: str) -> list[SearchResult]:
        if self.is_empty():
            return []
        escaped = path.replace("'", "''")
        rows = self.db.open_table(TABLE_NAME).search().where(f"path = '{escaped}'").to_list()
        rows.sort(key=lambda r: r["chunk_index"])
        return [
            SearchResult(
                id=row["id"],
                path=row["path"],
                heading=row["heading"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                score=1.0,
            )
            for row in rows
        ]

    def list_paths(self) -> list[str]:
        if self.is_empty():
            return []
        rows = self.db.open_table(TABLE_NAME).search().select(["path"]).to_list()
        return sorted({row["path"] for row in rows})
