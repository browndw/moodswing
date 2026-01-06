"""Tests for dictionary-based sentiment analyzer."""
from __future__ import annotations

from moodswing.sentiment.dictionary import DictionarySentimentAnalyzer

SAMPLES = [
    "I love sunny days but I hate humidity.",
    "Everything is neutral.",
]


def test_sentence_scores_default_method():
    analyzer = DictionarySentimentAnalyzer()
    scores = analyzer.text_scores(SAMPLES)
    assert len(scores) == 2
    assert scores[0] != 0


def test_nrc_emotions_positive_negative_keys():
    analyzer = DictionarySentimentAnalyzer()
    rows = analyzer.nrc_emotions(SAMPLES)
    assert len(rows) == 2
    assert "positive" in rows[0]
    assert "negative" in rows[0]


def test_mixed_messages_entropy_bounds():
    analyzer = DictionarySentimentAnalyzer()
    result = analyzer.mixed_messages(
        "I loved it but I hated it."
        )
    assert result.entropy >= 0
    assert result.normalized_entropy >= 0
