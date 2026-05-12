from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    vault_path: Path
    db_path: Path
    embedding_model: str = "all-MiniLM-L6-v2"
    chat_model: str = "claude-opus-5"
    effort: str = "medium"
    chunk_words: int = 220
    chunk_overlap_words: int = 40
    top_k: int = 6

    @classmethod
    def load(cls, vault_path: str | Path) -> Settings:
        vault = Path(vault_path).expanduser().resolve()
        db = Path(os.environ.get("VAULTMIND_DB", vault / ".vaultmind" / "index.lance"))
        return cls(
            vault_path=vault,
            db_path=db,
            embedding_model=os.environ.get("VAULTMIND_EMBEDDING_MODEL", cls.embedding_model),
            chat_model=os.environ.get("VAULTMIND_MODEL", cls.chat_model),
            effort=os.environ.get("VAULTMIND_EFFORT", cls.effort),
        )
