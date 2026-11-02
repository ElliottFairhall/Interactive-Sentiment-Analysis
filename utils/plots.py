"""Plotly figures for the sentiment analyser, in the Data Flakes style."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

POSITIVE = "#5E8FBA"
NEGATIVE = "#F67280"
NEUTRAL = "#CBD5E1"
FONT = {"family": "Inter, sans-serif", "size": 13}
GRID = "rgba(128,128,128,0.18)"
PREVIEW_CHARS = 90


def _style(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared layout: transparent, light grid, Inter text."""
    fig.update_layout(
        height=height,
        margin={"t": 8, "b": 8, "l": 8, "r": 8},
        font=FONT,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _colour(value: float) -> str:
    return POSITIVE if value > 0 else NEGATIVE if value < 0 else NEUTRAL


def _empty_figure(message: str) -> go.Figure:
    """A blank figure carrying one line of text."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, 240)


def draw_keyword_chart(word_counts: dict[str, int]) -> go.Figure:
    """Horizontal bars of the most frequent words.

    Args:
        word_counts: Words mapped to counts.

    Returns:
        The figure, or a placeholder if there are no words.
    """
    if not word_counts:
        return _empty_figure("No keywords found")
    words = list(word_counts)
    counts = list(word_counts.values())
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=words,
            orientation="h",
            marker_color=POSITIVE,
            text=counts,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    _style(fig, 60 + 32 * len(words))
    fig.update_layout(margin={"t": 8, "b": 8, "l": 8, "r": 32})
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Mentions", dtick=1)
    return fig


def draw_engine_comparison(textblob: float, vader: float) -> go.Figure:
    """Show both engines' polarity on one -1 to +1 axis.

    Args:
        textblob: TextBlob polarity.
        vader: VADER compound score.

    Returns:
        The figure.
    """
    names = ["TextBlob", "VADER"]
    values = [textblob, vader]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[_colour(v) for v in values],
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        )
    )
    _style(fig, 150)
    fig.update_layout(margin={"t": 8, "b": 8, "l": 8, "r": 40})
    fig.update_xaxes(range=[-1, 1], title="Negative  to  positive")
    fig.update_yaxes(autorange="reversed")
    return fig


def draw_sentence_timeline(scores: Sequence[Mapping[str, Any]]) -> go.Figure:
    """One bar per sentence, showing how the tone moves through the text.

    Args:
        scores: Sentence scores with ``text`` and ``polarity`` keys.

    Returns:
        The figure, or a placeholder if there are no sentences.
    """
    if not scores:
        return _empty_figure("No sentences to score")
    numbers = list(range(1, len(scores) + 1))
    values = [s["polarity"] for s in scores]
    previews = [
        s["text"]
        if len(s["text"]) <= PREVIEW_CHARS
        else s["text"][:PREVIEW_CHARS] + "..."
        for s in scores
    ]
    fig = go.Figure(
        go.Bar(
            x=numbers,
            y=values,
            marker_color=[_colour(v) for v in values],
            customdata=previews,
            hovertemplate="Sentence %{x}: %{y:+.2f}<br>%{customdata}<extra></extra>",
        )
    )
    _style(fig, 260)
    fig.update_xaxes(title="Sentence", dtick=1 if len(scores) <= 20 else None)
    fig.update_yaxes(range=[-1, 1], title="Polarity")
    return fig


def draw_word_valence(words: dict[str, float]) -> go.Figure:
    """Bars of the words that push the sentiment score up or down.

    Args:
        words: Words mapped to their sentiment valence.

    Returns:
        The figure, or a placeholder if no word carries sentiment.
    """
    if not words:
        return _empty_figure("No words with sentiment found")
    names = list(words)
    values = list(words.values())
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[_colour(v) for v in values],
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1f}<extra></extra>",
        )
    )
    _style(fig, 60 + 30 * len(names))
    fig.update_layout(margin={"t": 8, "b": 8, "l": 8, "r": 40})
    fig.update_xaxes(range=[-4.5, 4.5], title="Negative  to  positive")
    fig.update_yaxes(autorange="reversed")
    return fig
