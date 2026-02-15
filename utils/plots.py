"""Data visualization functions for sentiment analysis dashboard.

This module provides Plotly-based visualization functions for displaying
sentiment metrics, keyword frequencies, and other NLP analysis results.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Theme colors matching the dashboard aesthetic
COLORS: dict[str, str] = {
    "primary": "#fda4af",  # Rose 300
    "secondary": "#7dd3fc",  # Sky 300
    "background": "#0f172a",  # Slate 900
    "muted": "#1e293b",  # Slate 800
    "border": "#334155",  # Slate 700
    "positive": "rgba(34, 197, 94, 0.3)",  # Green
    "negative": "rgba(239, 68, 68, 0.3)",  # Red
    "neutral": "rgba(148, 163, 184, 0.3)",  # Gray
}


def draw_keyword_chart(word_counts: dict[str, int]) -> go.Figure:
    """Create a bar chart of keyword frequencies.

    Displays the most frequent words as a horizontal bar chart with
    gradient coloring based on frequency values.

    Args:
        word_counts: Dictionary mapping words to their frequency counts.

    Returns:
        Plotly Figure object with the keyword bar chart.

    Example:
        >>> counts = {"python": 10, "data": 8, "analysis": 5}
        >>> fig = draw_keyword_chart(counts)
        >>> fig.show()
    """
    if not word_counts:
        return _create_empty_figure("No keywords to display")

    df = pd.DataFrame(
        list(word_counts.items()),
        columns=["Keyword", "Frequency"],
    )
    df = df.sort_values(by="Frequency", ascending=False)

    fig = px.bar(
        df,
        x="Keyword",
        y="Frequency",
        color="Frequency",
        color_continuous_scale="Sunsetdark",
        template="plotly_dark",
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
        xaxis_tickangle=-45,
        font={"color": "white", "family": "Outfit"},
    )

    return fig


def draw_polarity_gauge(polarity: float) -> go.Figure:
    """Create a gauge chart displaying sentiment polarity.

    Visualizes the polarity score on a scale from -1 (negative) to 1 (positive)
    with color-coded zones indicating sentiment ranges.

    Args:
        polarity: Sentiment polarity score between -1 and 1.

    Returns:
        Plotly Figure object with the polarity gauge.

    Example:
        >>> fig = draw_polarity_gauge(0.75)
        >>> fig.show()  # Shows gauge indicating positive sentiment
    """
    # Clamp polarity to valid range
    polarity = max(-1.0, min(1.0, polarity))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=polarity,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Sentiment Polarity", "font": {"size": 18}},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": COLORS["primary"]},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [-1, -0.3], "color": COLORS["negative"]},
                    {"range": [-0.3, 0.3], "color": COLORS["neutral"]},
                    {"range": [0.3, 1], "color": COLORS["positive"]},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": polarity,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Outfit"},
        height=300,
    )

    return fig


def draw_subjectivity_pie(subjectivity: float | None) -> go.Figure | None:
    """Create a donut chart displaying subjectivity breakdown.

    Visualizes the ratio of subjective to objective content as a donut chart
    with the subjectivity score determining the split.

    Args:
        subjectivity: Subjectivity score between 0 (objective) and 1 (subjective).
            If None, returns None.

    Returns:
        Plotly Figure object with the subjectivity donut chart,
        or None if subjectivity is None.

    Example:
        >>> fig = draw_subjectivity_pie(0.6)
        >>> fig.show()  # Shows 60% subjective, 40% objective
    """
    if subjectivity is None:
        return None

    # Clamp subjectivity to valid range
    subjectivity = max(0.0, min(1.0, subjectivity))

    values = [subjectivity, 1 - subjectivity]
    labels = ["Subjective", "Objective"]

    fig = px.pie(
        values=values,
        names=labels,
        color_discrete_map={
            "Subjective": COLORS["secondary"],
            "Objective": COLORS["muted"],
        },
        hole=0.6,
        template="plotly_dark",
    )

    fig.update_layout(
        showlegend=False,
        annotations=[
            {
                "text": "Context",
                "x": 0.5,
                "y": 0.5,
                "font_size": 20,
                "showarrow": False,
            }
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 0, "b": 0, "l": 0, "r": 0},
        height=300,
    )

    return fig


def _create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a centered message.

    Args:
        message: Text to display in the center of the empty figure.

    Returns:
        Plotly Figure with the message annotation.
    """
    fig = go.Figure()

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Outfit"},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=300,
    )

    return fig
