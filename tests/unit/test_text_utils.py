"""Tests for text utilities."""
from __future__ import annotations

from pathlib import Path

from moodswing.text import Sentencizer, Tokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample_sentences.txt"


def test_tokenizer_basic_split():
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("Curly quotes \u2018hello\u2019!")
    assert tokens == ["curly", "quotes", "hello"]


def test_sentencizer_handles_sample_file():
    sentencizer = Sentencizer()
    sentences = sentencizer.split(FIXTURE.read_text(encoding="utf-8"))
    assert sentences[0] == "Mr. Brown comes!"
    assert "5 p.m. eastern time" in sentences[3]
    assert sentences[-1] == "Go there."
