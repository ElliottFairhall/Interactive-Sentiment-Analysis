"""Unit tests for the plots module."""

import plotly.graph_objects as go

from utils.plots import (
    draw_keyword_chart,
    draw_polarity_gauge,
    draw_subjectivity_pie,
)


class TestDrawKeywordChart:
    """Tests for keyword frequency bar chart."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        word_counts = {"hello": 5, "world": 3}
        result = draw_keyword_chart(word_counts)
        assert isinstance(result, go.Figure)

    def test_empty_dict_returns_figure(self) -> None:
        """Test that empty dictionary returns an empty figure."""
        result = draw_keyword_chart({})
        assert isinstance(result, go.Figure)

    def test_single_keyword(self) -> None:
        """Test that single keyword is handled correctly."""
        word_counts = {"python": 10}
        result = draw_keyword_chart(word_counts)
        assert isinstance(result, go.Figure)

    def test_many_keywords(self) -> None:
        """Test that many keywords are handled correctly."""
        word_counts = {f"word{i}": i for i in range(20)}
        result = draw_keyword_chart(word_counts)
        assert isinstance(result, go.Figure)


class TestDrawPolarityGauge:
    """Tests for polarity gauge chart."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        result = draw_polarity_gauge(0.5)
        assert isinstance(result, go.Figure)

    def test_positive_polarity(self) -> None:
        """Test gauge with positive polarity."""
        result = draw_polarity_gauge(0.8)
        assert isinstance(result, go.Figure)

    def test_negative_polarity(self) -> None:
        """Test gauge with negative polarity."""
        result = draw_polarity_gauge(-0.8)
        assert isinstance(result, go.Figure)

    def test_neutral_polarity(self) -> None:
        """Test gauge with neutral polarity."""
        result = draw_polarity_gauge(0.0)
        assert isinstance(result, go.Figure)

    def test_extreme_positive(self) -> None:
        """Test gauge at maximum positive value."""
        result = draw_polarity_gauge(1.0)
        assert isinstance(result, go.Figure)

    def test_extreme_negative(self) -> None:
        """Test gauge at maximum negative value."""
        result = draw_polarity_gauge(-1.0)
        assert isinstance(result, go.Figure)

    def test_clamps_out_of_range_high(self) -> None:
        """Test that values above 1 are clamped."""
        result = draw_polarity_gauge(1.5)
        assert isinstance(result, go.Figure)

    def test_clamps_out_of_range_low(self) -> None:
        """Test that values below -1 are clamped."""
        result = draw_polarity_gauge(-1.5)
        assert isinstance(result, go.Figure)


class TestDrawSubjectivityPie:
    """Tests for subjectivity pie chart."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        result = draw_subjectivity_pie(0.5)
        assert isinstance(result, go.Figure)

    def test_none_returns_none(self) -> None:
        """Test that None input returns None."""
        result = draw_subjectivity_pie(None)
        assert result is None

    def test_fully_subjective(self) -> None:
        """Test pie chart with fully subjective score."""
        result = draw_subjectivity_pie(1.0)
        assert isinstance(result, go.Figure)

    def test_fully_objective(self) -> None:
        """Test pie chart with fully objective score."""
        result = draw_subjectivity_pie(0.0)
        assert isinstance(result, go.Figure)

    def test_half_subjective(self) -> None:
        """Test pie chart with 50/50 split."""
        result = draw_subjectivity_pie(0.5)
        assert isinstance(result, go.Figure)

    def test_clamps_out_of_range_high(self) -> None:
        """Test that values above 1 are clamped."""
        result = draw_subjectivity_pie(1.5)
        assert isinstance(result, go.Figure)

    def test_clamps_out_of_range_low(self) -> None:
        """Test that values below 0 are clamped."""
        result = draw_subjectivity_pie(-0.5)
        assert isinstance(result, go.Figure)
