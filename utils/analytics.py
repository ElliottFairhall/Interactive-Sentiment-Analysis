"""Sentiment analysis and NLP utility functions.

This module provides functions for sentiment analysis using TextBlob and VADER,
keyword extraction, and named entity recognition using spaCy.
"""

import logging
import re
from collections import Counter
from typing import TypedDict

import nltk
import spacy
from nltk.corpus import stopwords
from spacy.language import Language
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


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


def _load_spacy_model() -> Language:
    """Load the spaCy English model, downloading if necessary.

    Returns:
        Loaded spaCy Language model.

    Raises:
        OSError: If the model cannot be loaded or downloaded.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy en_core_web_sm model...")
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


# Initialize NLP resources
nlp = _load_spacy_model()
nltk.download("stopwords", quiet=True)
STOP_WORDS: set[str] = set(stopwords.words("english"))


def analyze_sentiment_textblob(text: str) -> TextBlobResult:
    """Analyze sentiment using TextBlob.

    TextBlob provides both polarity (emotional lean) and subjectivity
    (factual vs opinionated) scores for text analysis.

    Args:
        text: The text to analyze.

    Returns:
        Dictionary containing:
            - polarity: Float from -1 (negative) to 1 (positive)
            - subjectivity: Float from 0 (objective) to 1 (subjective)
            - sentiment: String classification (Positive, Neutral, Negative)

    Example:
        >>> result = analyze_sentiment_textblob("I love this product!")
        >>> print(result['sentiment'])
        Positive
    """
    if not text or not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0, "sentiment": "Neutral"}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment,
    }


def analyze_sentiment_vader(text: str) -> VaderResult:
    """Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER is particularly effective for social media text and provides
    detailed sentiment scores including compound, positive, negative, and neutral.

    Args:
        text: The text to analyze.

    Returns:
        Dictionary containing:
            - polarity: Float from -1 to 1 (compound score)
            - subjectivity: None (VADER doesn't provide subjectivity)
            - sentiment: String classification (Positive, Neutral, Negative)
            - raw_scores: Dictionary with pos, neg, neu, and compound scores

    Example:
        >>> result = analyze_sentiment_vader("This is absolutely amazing!")
        >>> print(result['sentiment'])
        Positive
    """
    if not text or not text.strip():
        return {
            "polarity": 0.0,
            "subjectivity": None,
            "sentiment": "Neutral",
            "raw_scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0},
        }

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
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
    """Extract the most frequent words from text, excluding stopwords.

    Tokenizes text and counts word frequencies, filtering out common
    English stopwords and single-character tokens.

    Args:
        text: The text to analyze.
        n_top: Number of top keywords to return. Defaults to 10.

    Returns:
        Dictionary mapping words to their frequency counts,
        sorted by frequency in descending order.

    Example:
        >>> freq = get_word_frequency("The quick brown fox jumps over the lazy dog")
        >>> print(list(freq.keys())[:3])
        ['quick', 'brown', 'fox']
    """
    if not text or not text.strip():
        return {}

    words = re.findall(r"\b\w+\b", text.lower())
    words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    word_counts = Counter(words)
    return dict(word_counts.most_common(n_top))


def get_named_entities(text: str) -> list[EntityResult]:
    """Extract named entities from text using spaCy NER.

    Identifies and classifies named entities such as people, organizations,
    locations, dates, and other categories.

    Args:
        text: The text to analyze.

    Returns:
        List of dictionaries, each containing:
            - text: The entity text
            - label: The entity type (PERSON, ORG, GPE, DATE, etc.)

    Example:
        >>> entities = get_named_entities("Apple Inc. was founded by Steve Jobs.")
        >>> print([e['label'] for e in entities])
        ['ORG', 'PERSON']
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)
    entities: list[EntityResult] = []

    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    return entities
