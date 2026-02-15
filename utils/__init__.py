"""Utility modules for the Sentiment Dynamics dashboard.

This package provides:
- analytics: Sentiment analysis, keyword extraction, and NER functions
- plots: Plotly-based visualization functions
- text: UI text content and documentation
"""

from utils.analytics import (
    analyze_sentiment_textblob,
    analyze_sentiment_vader,
    get_named_entities,
    get_word_frequency,
)
from utils.plots import (
    draw_keyword_chart,
    draw_polarity_gauge,
    draw_subjectivity_pie,
)
from utils.text import (
    analysis_guide,
    header_info,
    keyword_explained,
    ner_explained,
    polarity_explained,
    project_overview,
    sentiment_definitions,
    subjectivity_explained,
)

__all__ = [
    # Analytics
    "analyze_sentiment_textblob",
    "analyze_sentiment_vader",
    "get_word_frequency",
    "get_named_entities",
    # Plots
    "draw_keyword_chart",
    "draw_polarity_gauge",
    "draw_subjectivity_pie",
    # Text
    "header_info",
    "project_overview",
    "analysis_guide",
    "sentiment_definitions",
    "polarity_explained",
    "subjectivity_explained",
    "keyword_explained",
    "ner_explained",
]
