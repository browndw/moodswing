"""Tests for packaged data helpers."""
from __future__ import annotations

from moodswing.data import (
    iter_sample_texts,
    load_sample_text,
    load_text_directory,
    load_text_file,
)


def test_iter_sample_texts_yields_documents() -> None:
    items = list(iter_sample_texts())
    assert items, "expected at least one sample text"
    for doc_id, text in items:
        assert doc_id
        assert text


def test_load_sample_text_defaults_to_first_entry() -> None:
    doc_id, text = load_sample_text()
    assert isinstance(doc_id, str) and doc_id
    assert isinstance(text, str) and text


def test_load_text_file_normalizes_content(tmp_path) -> None:
    payload = ' Hello\tWorld!\r\n“Curly” café \n\n'
    source = tmp_path / "My Story.TXT"
    source.write_text(payload, encoding="utf-8")

    record = load_text_file(source)

    assert record["doc_id"] == "my_story"
    assert "cafe" in record["text"]
    assert "\t" not in record["text"]
    assert "\r" not in record["text"]


def test_load_text_directory_reads_top_level_txt_files(tmp_path) -> None:
    base = tmp_path / "texts"
    base.mkdir()
    (base / "a.txt").write_text("Alpha", encoding="utf-8")
    (base / "b.txt").write_text("Beta", encoding="utf-8")
    sub = base / "nested"
    sub.mkdir()
    (sub / "ignored.txt").write_text("Skip", encoding="utf-8")

    records = load_text_directory(base)

    assert [rec["doc_id"] for rec in records] == ["a", "b"]
    assert all(rec["text"] for rec in records)
