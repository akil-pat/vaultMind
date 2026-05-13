from pathlib import Path

from vaultmind.ingest.chunker import chunk_document
from vaultmind.ingest.loader import Document


def test_splits_on_markdown_headers():
    text = "# Intro\nfirst section body\n\n## Details\nsecond section body\n"
    doc = Document(path=Path("notes.md"), text=text)

    chunks = chunk_document(doc, chunk_words=50, overlap_words=10)

    assert [c.heading for c in chunks] == ["Intro", "Details"]
    assert "first section body" in chunks[0].text
    assert "second section body" in chunks[1].text


def test_preamble_before_first_header_is_kept():
    text = "no heading yet\n\n# Real Section\nbody text\n"
    doc = Document(path=Path("notes.md"), text=text)

    chunks = chunk_document(doc, chunk_words=50, overlap_words=10)

    assert chunks[0].heading == "(untitled)"
    assert "no heading yet" in chunks[0].text
    assert chunks[1].heading == "Real Section"


def test_long_section_splits_into_overlapping_windows():
    words = [f"word{i}" for i in range(100)]
    text = "# Long Section\n" + " ".join(words) + "\n"
    doc = Document(path=Path("notes.md"), text=text)

    chunks = chunk_document(doc, chunk_words=30, overlap_words=10)

    assert len(chunks) > 1
    # consecutive windows overlap: the tail of one chunk reappears at the head of the next
    first_tail = chunks[0].text.split()[-10:]
    second_head = chunks[1].text.split()[:10]
    assert first_tail == second_head


def test_non_markdown_file_is_single_untitled_section():
    doc = Document(path=Path("notes.txt"), text="just plain text, no headers here")

    chunks = chunk_document(doc, chunk_words=50, overlap_words=10)

    assert len(chunks) == 1
    assert chunks[0].heading == "notes"


def test_chunk_ids_are_unique_and_sequential():
    words = [f"word{i}" for i in range(80)]
    text = "# A\n" + " ".join(words) + "\n# B\nshort body\n"
    doc = Document(path=Path("notes.md"), text=text)

    chunks = chunk_document(doc, chunk_words=20, overlap_words=5)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
