from __future__ import annotations

import typer
from rich.console import Console

from vaultmind.agent import Agent
from vaultmind.config import Settings
from vaultmind.ingest.chunker import chunk_document
from vaultmind.ingest.embedder import Embedder
from vaultmind.ingest.loader import load_vault
from vaultmind.ingest.store import VaultStore

app = typer.Typer(help="vaultmind — an agentic RAG assistant for your own notes.")
console = Console()


@app.command()
def ingest(vault: str = typer.Argument(..., help="Path to your notes folder")) -> None:
    """Index (or re-index) a vault of Markdown/text/PDF notes for search."""
    settings = Settings.load(vault)
    docs = list(load_vault(settings.vault_path))
    if not docs:
        console.print(f"[yellow]No Markdown/text/PDF files found under {settings.vault_path}[/yellow]")
        raise typer.Exit(code=1)

    embedder = Embedder(settings.embedding_model)
    records = []
    with console.status(f"Embedding {len(docs)} note(s)..."):
        for doc in docs:
            chunks = chunk_document(doc, settings.chunk_words, settings.chunk_overlap_words)
            if not chunks:
                continue
            vectors = embedder.embed([c.text for c in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
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

    store = VaultStore(settings.db_path)
    store.rebuild(records)
    console.print(f"[green]Indexed {len(docs)} note(s) into {len(records)} chunk(s).[/green]")
    console.print(f"Index stored at {settings.db_path}")


def _require_index(settings: Settings, vault_arg: str) -> None:
    store = VaultStore(settings.db_path)
    if store.is_empty():
        console.print(
            f"[yellow]No index found for {settings.vault_path} — "
            f"run `vaultmind ingest {vault_arg}` first.[/yellow]"
        )
        raise typer.Exit(code=1)


@app.command()
def chat(vault: str = typer.Argument(..., help="Path to your notes folder")) -> None:
    """Start an interactive chat session with your notes."""
    settings = Settings.load(vault)
    _require_index(settings, vault)

    agent = Agent(settings)
    console.print(
        f"[bold]vaultmind[/bold] — chatting with {settings.vault_path} "
        f"(model: {settings.chat_model}). Type 'exit' to quit.\n"
    )

    history: list[dict] = []
    while True:
        try:
            question = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        if not question.strip():
            continue

        def on_tool_call(name: str, tool_input: dict) -> None:
            arg = next(iter(tool_input.values()), "")
            console.print(f"[dim]  -> {name}({arg!r})[/dim]")

        with console.status("thinking..."):
            answer, history = agent.ask(question, history, on_tool_call=on_tool_call)
        console.print(f"[bold magenta]vaultmind>[/bold magenta] {answer}\n")


@app.command()
def ask(
    vault: str = typer.Argument(..., help="Path to your notes folder"),
    question: str = typer.Argument(..., help="A single question to ask"),
) -> None:
    """Ask a single one-shot question — handy for scripting or demos."""
    settings = Settings.load(vault)
    _require_index(settings, vault)

    agent = Agent(settings)
    answer, _ = agent.ask(question)
    console.print(answer)


if __name__ == "__main__":
    app()
