"""Behaviour tests for the new analysis helpers and the live page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.analytics import (
    MAX_CHARS,
    SentenceScore,
    compare_engines,
    get_nlp,
    get_stop_words,
    score_sentences,
    sentiment_words,
    split_sentences,
    text_stats,
)
from utils.layout import EXAMPLES, highlight_html

ROOT = Path(__file__).resolve().parent.parent
REVIEW = EXAMPLES["Product review"]
COMPLAINT = EXAMPLES["Customer complaint"]


def test_models_are_cached_not_reloaded() -> None:
    """The spaCy pipeline and stop words load once and are then reused."""
    assert get_nlp() is get_nlp()
    assert get_stop_words() is get_stop_words()


def test_split_sentences_finds_each_sentence() -> None:
    """The product review has four sentences."""
    assert len(split_sentences(REVIEW)) == 4
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_sentence_scores_follow_the_tone() -> None:
    """The downside sentence of the review is the negative one."""
    scores = score_sentences(REVIEW, "VADER")
    assert [s["sentiment"] for s in scores] == [
        "Positive",
        "Positive",
        "Negative",
        "Positive",
    ]


def test_both_engines_can_score_sentences() -> None:
    """The engine argument switches the method without changing the shape."""
    vader = score_sentences(REVIEW, "VADER")
    blob = score_sentences(REVIEW, "TextBlob")
    assert len(vader) == len(blob) == 4
    assert vader != blob


def test_complaint_reads_negative_in_both_engines() -> None:
    """A clear complaint is negative whichever method is used."""
    result = compare_engines(COMPLAINT)
    assert result["vader"]["sentiment"] == "Negative"
    assert result["textblob"]["sentiment"] == "Negative"
    assert result["agree"] is True


def test_compare_engines_reports_disagreement() -> None:
    """When labels differ, agree is false."""
    result = compare_engines("It is fine.")
    assert result["agree"] == (
        result["textblob"]["sentiment"] == result["vader"]["sentiment"]
    )


def test_sentiment_words_name_the_drivers() -> None:
    """Love is positive and awful is negative, each listed once."""
    words = sentiment_words("I love it but the service was awful. Love it!")
    assert words["love"] > 0
    assert words["awful"] < 0
    assert list(words).count("love") == 1


def test_sentiment_words_is_empty_for_neutral_text() -> None:
    """Text without sentiment words gives nothing, not an error."""
    assert sentiment_words("The table is in the room.") == {}


def test_text_stats_counts_words_and_sentences() -> None:
    """Simple counts on a known sentence pair."""
    stats = text_stats("One two three. Four five six seven.")
    assert stats["words"] == 7
    assert stats["sentences"] == 2
    assert stats["words_per_sentence"] == pytest.approx(3.5)


def test_text_stats_handles_empty_text() -> None:
    """No text means zeros, never a division error."""
    stats = text_stats("")
    assert stats["words"] == 0
    assert stats["words_per_sentence"] == 0.0
    assert stats["unique_share"] == 0.0


def test_highlight_escapes_html_in_user_text() -> None:
    """Markup typed by a user is shown as text, never run."""
    scores: list[SentenceScore] = [
        {
            "text": '<script>alert("x")</script> great',
            "polarity": 0.6,
            "sentiment": "Positive",
        }
    ]
    markup = highlight_html(scores)
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


def _run() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


def test_empty_page_invites_the_user_to_start() -> None:
    """With no text the page explains what to do."""
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("pick an example" in info.value for info in at.info)


def test_text_produces_both_engine_results() -> None:
    """Typing a review shows VADER, TextBlob and subjectivity."""
    at = _run()
    at.text_area(key="text").set_value(REVIEW).run()
    assert not at.exception
    labels = [m.label for m in at.metric if m.label.startswith(("VADER", "TextBlob"))]
    assert len(labels) == 2
    assert any(m.label == "Subjectivity" for m in at.metric)


def test_choosing_an_example_fills_the_text_box() -> None:
    """The example buttons load their text and the analysis follows."""
    at = _run()
    next(
        c
        for c in at.get("segmented_control")
        if getattr(c, "label", "") == "Try an example"
    ).set_value("News report").run()
    assert not at.exception
    assert at.text_area(key="text").value == EXAMPLES["News report"]
    assert any(m.label.startswith("VADER") for m in at.metric)


def test_short_text_is_not_analysed() -> None:
    """A few characters is below the minimum, so the prompt stays."""
    at = _run()
    at.text_area(key="text").set_value("ok").run()
    assert not at.exception
    assert not any(m.label.startswith("VADER") for m in at.metric)


def test_hostile_markup_does_not_break_the_page() -> None:
    """HTML in the input is analysed as text without raising."""
    at = _run()
    at.text_area(key="text").set_value(
        "<img src=x onerror=alert(1)> I love this, it is great."
    ).run()
    assert not at.exception


def test_text_over_the_limit_is_trimmed_with_a_warning() -> None:
    """Overlong input is cut to the limit and the user is told."""
    at = _run()
    at.text_area(key="text").set_value("Good. " * (MAX_CHARS // 4)).run()
    assert not at.exception
    assert any(str(MAX_CHARS) in w.value.replace(",", "") for w in at.warning)


def test_switching_the_sentence_engine_renders() -> None:
    """TextBlob sentence scoring works as well as VADER."""
    at = _run()
    at.text_area(key="text").set_value(REVIEW).run()
    next(
        c
        for c in at.get("segmented_control")
        if getattr(c, "label", "") == "Score sentences with"
    ).set_value("TextBlob").run()
    assert not at.exception


def test_tiny_textblob_scores_are_neutral_not_positive() -> None:
    """A score of +0.00 must not be labelled Positive (same band as VADER)."""
    from utils.analytics import analyze_sentiment_textblob

    result = analyze_sentiment_textblob(EXAMPLES["News report"])
    assert abs(result["polarity"]) < 0.05
    assert result["sentiment"] == "Neutral"


def test_entity_groups_have_readable_names() -> None:
    """SpaCy codes such as CARDINAL are shown as plain words."""
    from utils.analytics import ENTITY_LABELS

    assert ENTITY_LABELS["CARDINAL"] == "Numbers"
    assert all(label[0].isupper() for label in ENTITY_LABELS.values())
