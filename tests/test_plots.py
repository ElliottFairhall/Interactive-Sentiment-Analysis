"""Unit tests for the plots module."""

import plotly.graph_objects as go
import pytest

from utils.plots import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    draw_engine_comparison,
    draw_keyword_chart,
    draw_sentence_timeline,
    draw_word_valence,
)


class TestDrawKeywordChart:
    """Tests for the keyword frequency bar chart."""

    def test_returns_figure(self) -> None:
        """A normal dictionary gives a figure with one bar per word."""
        figure = draw_keyword_chart({"hello": 5, "world": 3})
        assert isinstance(figure, go.Figure)
        assert list(figure.data[0].y) == ["hello", "world"]

    def test_empty_dict_returns_placeholder(self) -> None:
        """No words gives a placeholder figure, not an error."""
        figure = draw_keyword_chart({})
        assert isinstance(figure, go.Figure)
        assert figure.layout.annotations[0].text == "No keywords found"

    def test_single_keyword(self) -> None:
        """One word is fine."""
        assert isinstance(draw_keyword_chart({"python": 10}), go.Figure)


class TestDrawEngineComparison:
    """Tests for the two-engine polarity bars."""

    def test_colours_follow_the_sign(self) -> None:
        """Positive is blue, negative is rose."""
        figure = draw_engine_comparison(-0.4, 0.6)
        assert list(figure.data[0].marker.color) == [NEGATIVE, POSITIVE]

    def test_axis_covers_minus_one_to_one(self) -> None:
        """The scale is fixed so the two engines are comparable."""
        figure = draw_engine_comparison(0.0, 0.0)
        assert tuple(figure.layout.xaxis.range) == (-1, 1)
        assert figure.data[0].marker.color[0] == NEUTRAL


class TestDrawSentenceTimeline:
    """Tests for the per-sentence bars."""

    def test_one_bar_per_sentence(self) -> None:
        """Bars are numbered from 1 and coloured by sign."""
        scores = [
            {"text": "Great.", "polarity": 0.8},
            {"text": "Awful.", "polarity": -0.7},
            {"text": "Fine.", "polarity": 0.0},
        ]
        figure = draw_sentence_timeline(scores)
        assert list(figure.data[0].x) == [1, 2, 3]
        assert list(figure.data[0].marker.color) == [POSITIVE, NEGATIVE, NEUTRAL]

    def test_long_sentences_are_shortened_in_the_hover(self) -> None:
        """Hover previews never run to hundreds of characters."""
        figure = draw_sentence_timeline([{"text": "word " * 100, "polarity": 0.1}])
        assert len(figure.data[0].customdata[0]) < 100

    def test_empty_input_returns_placeholder(self) -> None:
        """No sentences gives a placeholder."""
        figure = draw_sentence_timeline([])
        assert figure.layout.annotations[0].text == "No sentences to score"


class TestDrawWordValence:
    """Tests for the words-that-move-the-score chart."""

    def test_bars_are_coloured_by_valence(self) -> None:
        """Positive words blue, negative words rose."""
        figure = draw_word_valence({"love": 3.2, "awful": -2.0})
        assert list(figure.data[0].marker.color) == [POSITIVE, NEGATIVE]

    @pytest.mark.parametrize("words", [{}])
    def test_no_sentiment_words_returns_placeholder(self, words: dict) -> None:
        """A text with no sentiment words gives a placeholder."""
        figure = draw_word_valence(words)
        assert figure.layout.annotations[0].text == "No words with sentiment found"
