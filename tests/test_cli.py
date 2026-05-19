from pathlib import Path

from typer.testing import CliRunner

from vaultmind import cli

runner = CliRunner()


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Topic\n\nSome content about the topic.\n")
    return vault


def test_ingest_creates_index(tmp_path: Path):
    vault = _make_vault(tmp_path)

    result = runner.invoke(cli.app, ["ingest", str(vault)])

    assert result.exit_code == 0, result.output
    assert "Indexed 1 note(s)" in result.output
    assert (vault / ".vaultmind" / "index.lance").exists()


def test_ingest_empty_vault_exits_nonzero(tmp_path: Path):
    empty_vault = tmp_path / "empty"
    empty_vault.mkdir()

    result = runner.invoke(cli.app, ["ingest", str(empty_vault)])

    assert result.exit_code == 1
    assert "No Markdown/text/PDF files found" in result.output


def test_ask_without_index_exits_nonzero(tmp_path: Path):
    vault = tmp_path / "unindexed"
    vault.mkdir()

    result = runner.invoke(cli.app, ["ask", str(vault), "anything?"])

    assert result.exit_code == 1
    assert "No index found" in result.output


def test_ask_prints_agent_answer(tmp_path: Path, monkeypatch):
    vault = _make_vault(tmp_path)
    runner.invoke(cli.app, ["ingest", str(vault)])

    class StubAgent:
        def __init__(self, settings):
            self.settings = settings

        def ask(self, question, history=None, on_tool_call=None):
            return f"stub answer to: {question}", []

    monkeypatch.setattr(cli, "Agent", StubAgent)

    result = runner.invoke(cli.app, ["ask", str(vault), "what's in my notes?"])

    assert result.exit_code == 0
    assert "stub answer to: what's in my notes?" in result.output
