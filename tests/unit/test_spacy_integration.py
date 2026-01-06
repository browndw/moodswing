"""Tests for the spaCy-backed helpers."""
from __future__ import annotations

import spacy
from spacy.language import Language

from moodswing.sentiment import SpaCySentimentAnalyzer
from moodswing.text import SpaCySentencizer


def test_spacy_sentencizer_splits_text() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    splitter = SpaCySentencizer(nlp=nlp)
    sentences = splitter.split("One. Two?")
    assert sentences == ["One.", "Two?"]


def test_spacy_sentiment_analyzer_uses_doc_cats() -> None:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    @Language.component("toy_sentiment")
    def toy_sentiment(doc):  # type: ignore[override]
        text = doc.text.lower()
        doc.cats["POSITIVE"] = 1.0 if "love" in text else 0.0
        doc.cats["NEGATIVE"] = 1.0 if "hate" in text else 0.0
        return doc

    nlp.add_pipe("toy_sentiment")

    analyzer = SpaCySentimentAnalyzer(nlp=nlp)
    sentences = ["I love cake.", "I hate rain."]
    scores = analyzer.sentence_scores(sentences)
    assert scores[0] > 0
    assert scores[1] < 0

    text_scores = analyzer.text_scores("I love cake. I hate rain.")
    assert len(text_scores) == 2
    assert text_scores[0] > text_scores[1]
