"""Sentiment analysis and NLP helpers.

Sentiment comes from TextBlob and VADER, keywords from simple word counts and
named entities from spaCy. The spaCy model and NLTK stop words load on first
use, not on import, so importing this module is fast and never touches the
network.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from functools import lru_cache
from typing import TypedDict

import nltk
import spacy
from nltk.corpus import stopwords
from spacy.language import Language
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

MAX_CHARS = 20_000
VADER_BAND = 0.05
ENTITY_LABELS = {
    "PERSON": "People",
    "ORG": "Organisations",
    "GPE": "Places",
    "LOC": "Locations",
    "DATE": "Dates",
    "TIME": "Times",
    "MONEY": "Money",
    "PERCENT": "Percentages",
    "PRODUCT": "Products",
    "EVENT": "Events",
    "NORP": "Groups",
    "CARDINAL": "Numbers",
    "ORDINAL": "Ordinals",
    "QUANTITY": "Quantities",
    "FAC": "Buildings and sites",
    "WORK_OF_ART": "Works of art",
    "LAW": "Laws",
    "LANGUAGE": "Languages",
}


class TextBlobResult(TypedDict):
    """Result from TextBlob sentiment analysis."""

    polarity: float
    subjectivity: float
    sentiment: str


class VaderResult(TypedDict):
    """Result from VADER sentiment analysis."""

    polarity: float
    subjectivity: None
    sentiment: str
    raw_scores: dict[str, float]


class EntityResult(TypedDict):
    """Named entity recognition result."""

    text: str
    label: str


class SentenceScore(TypedDict):
    """Sentiment of a single sentence."""

    text: str
    polarity: float
    sentiment: str


class EngineComparison(TypedDict):
    """Both engines' results and whether their labels match."""

    textblob: TextBlobResult
    vader: VaderResult
    agree: bool


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Load the spaCy English model once, downloading it only if missing.

    The model is normally installed as a dependency (see requirements.txt);
    the download is a fallback for environments that skipped it.

    Returns:
        The loaded spaCy pipeline.

    Raises:
        OSError: If the model is missing and cannot be downloaded.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy en_core_web_sm model...")
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def get_stop_words() -> frozenset[str]:
    """Load NLTK's English stop words once, fetching them if missing.

    Returns:
        The set of stop words.
    """
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))


def analyze_sentiment_textblob(text: str) -> TextBlobResult:
    """Analyze sentiment using TextBlob.

    TextBlob gives both polarity (emotional lean) and subjectivity (factual
    versus opinionated). Scores within 0.05 of zero count as Neutral, the
    same band VADER uses, so the two labels can be compared.

    Args:
        text: The text to analyze.

    Returns:
        Polarity from -1 to 1, subjectivity from 0 to 1 and a label.

    Example:
        >>> analyze_sentiment_textblob("I love this product!")["sentiment"]
        'Positive'
    """
    if not text or not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0, "sentiment": "Neutral"}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity >= VADER_BAND:
        sentiment = "Positive"
    elif polarity <= -VADER_BAND:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment,
    }


def analyze_sentiment_vader(text: str) -> VaderResult:
    """Analyze sentiment using VADER.

    VADER is tuned for short, informal text and returns a compound score plus
    the positive, negative and neutral shares.

    Args:
        text: The text to analyze.

    Returns:
        The compound score as ``polarity``, a label and the raw scores.

    Example:
        >>> analyze_sentiment_vader("This is absolutely amazing!")["sentiment"]
        'Positive'
    """
    if not text or not text.strip():
        return {
            "polarity": 0.0,
            "subjectivity": None,
            "sentiment": "Neutral",
            "raw_scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0},
        }

    scores = SentimentIntensityAnalyzer().polarity_scores(text)
    compound = scores["compound"]

    if compound >= VADER_BAND:
        sentiment = "Positive"
    elif compound <= -VADER_BAND:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "polarity": compound,
        "subjectivity": None,
        "sentiment": sentiment,
        "raw_scores": scores,
    }


def get_word_frequency(text: str, n_top: int = 10) -> dict[str, int]:
    """Count the most frequent words, ignoring stop words and single letters.

    Args:
        text: The text to analyze.
        n_top: Number of top words to return.

    Returns:
        Words mapped to counts, most frequent first.

    Example:
        >>> list(get_word_frequency("The quick brown fox jumps over the lazy dog"))[:2]
        ['quick', 'brown']
    """
    if not text or not text.strip():
        return {}

    stop_words = get_stop_words()
    words = re.findall(r"\b\w+\b", text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return dict(Counter(words).most_common(n_top))


def get_named_entities(text: str) -> list[EntityResult]:
    """Extract named entities with spaCy.

    Args:
        text: The text to analyze.

    Returns:
        One dict per entity with its ``text`` and ``label`` (PERSON, ORG, GPE...).

    Example:
        >>> found = get_named_entities("Apple Inc. was founded by Steve Jobs.")
        >>> [e["label"] for e in found]
        ['ORG', 'PERSON']
    """
    if not text or not text.strip():
        return []
    return [{"text": ent.text, "label": ent.label_} for ent in get_nlp()(text).ents]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with spaCy.

    Args:
        text: The text to split.

    Returns:
        Non-empty, stripped sentences in order.
    """
    if not text or not text.strip():
        return []
    return [s.text.strip() for s in get_nlp()(text).sents if s.text.strip()]


def score_sentences(text: str, engine: str = "VADER") -> list[SentenceScore]:
    """Score every sentence, to show where the tone shifts.

    Args:
        text: The text to analyze.
        engine: ``"VADER"`` or ``"TextBlob"``.

    Returns:
        One score per sentence, in reading order.
    """
    analyse = (
        analyze_sentiment_vader if engine == "VADER" else analyze_sentiment_textblob
    )
    scored: list[SentenceScore] = []
    for sentence in split_sentences(text):
        result = analyse(sentence)
        scored.append(
            {
                "text": sentence,
                "polarity": result["polarity"],
                "sentiment": result["sentiment"],
            }
        )
    return scored


def compare_engines(text: str) -> EngineComparison:
    """Run both engines and say whether they agree.

    Args:
        text: The text to analyze.

    Returns:
        ``textblob`` and ``vader`` results plus ``agree``, true when both give
        the same label.
    """
    blob = analyze_sentiment_textblob(text)
    vader = analyze_sentiment_vader(text)
    return {
        "textblob": blob,
        "vader": vader,
        "agree": blob["sentiment"] == vader["sentiment"],
    }


def text_stats(text: str) -> dict[str, float]:
    """Basic size and readability figures.

    Args:
        text: The text to measure.

    Returns:
        ``words``, ``sentences``, ``words_per_sentence`` and ``unique_share``
        (the fraction of words that are distinct).
    """
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = len(split_sentences(text))
    return {
        "words": len(words),
        "sentences": sentences,
        "words_per_sentence": len(words) / sentences if sentences else 0.0,
        "unique_share": len(set(words)) / len(words) if words else 0.0,
    }


def sentiment_words(text: str, limit: int = 12) -> dict[str, float]:
    """Find the words that push VADER's score up or down.

    Looks each word up in VADER's sentiment dictionary, so this shows why a
    text scored the way it did. Negation and intensifiers are not applied here;
    it is a reading aid, not the score itself.

    Args:
        text: The text to analyze.
        limit: Maximum number of words to return.

    Returns:
        Words mapped to their dictionary valence (about -4 to +4), strongest
        first, each word listed once.
    """
    lexicon = SentimentIntensityAnalyzer().lexicon
    found: dict[str, float] = {}
    for token in re.findall(r"[\w']+", text.lower()):
        valence = lexicon.get(token, 0.0)
        if valence and token not in found:
            found[token] = float(valence)
    ranked = sorted(found.items(), key=lambda item: abs(item[1]), reverse=True)
    return dict(ranked[:limit])
