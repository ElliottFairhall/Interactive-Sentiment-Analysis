"""Page layout for the sentiment analyser."""

from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path

import streamlit as st

from utils.analytics import (
    ENTITY_LABELS,
    MAX_CHARS,
    SentenceScore,
    compare_engines,
    get_named_entities,
    get_word_frequency,
    score_sentences,
    sentiment_words,
    text_stats,
)
from utils.plots import (
    draw_engine_comparison,
    draw_keyword_chart,
    draw_sentence_timeline,
    draw_word_valence,
)

SITE_URL = "https://www.data-flakes.dev"
ROOT = Path(__file__).resolve().parent.parent
MIN_CHARS = 5
POSITIVE_RGB = "94,143,186"
NEGATIVE_RGB = "246,114,128"

EXAMPLES = {
    "Product review": (
        "I bought this blender last month and I absolutely love it. It crushes ice "
        "in seconds and it is easy to clean. The only downside is the noise, which "
        "is quite loud. Overall it is great value for the price."
    ),
    "Customer complaint": (
        "The delivery was two weeks late and nobody answered my emails. When the "
        "parcel finally arrived the box was damaged and one item was missing. I am "
        "very disappointed and I want a refund."
    ),
    "News report": (
        "Northwind Bank announced on Tuesday that it will open a new research "
        "centre in Manchester, creating 500 jobs. The chief executive said the "
        "investment reflects confidence in the UK economy. The centre is due to "
        "open in March 2027."
    ),
}


def _load_example() -> None:
    """Copy the chosen example into the text box."""
    choice = st.session_state.get("example")
    if choice:
        st.session_state["text"] = EXAMPLES[choice]


def _load_upload() -> None:
    """Read an uploaded text file into the text box."""
    uploaded = st.session_state.get("upload")
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8", errors="replace")
        st.session_state["text"] = raw[:MAX_CHARS]
        st.session_state["example"] = None


def render_config() -> None:
    """Set the page configuration and load the stylesheet."""
    st.set_page_config(
        page_title="Text sentiment analyser",
        page_icon=str(ROOT / "assets" / "images" / "favicon.png"),
        layout="wide",
    )
    css = (ROOT / "styles" / "main.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_topbar() -> None:
    """Show a plain breadcrumb back to the Data Flakes site."""
    st.markdown(
        f"""
        <nav class="site-bar">
            <a href="{SITE_URL}/" target="_blank">Data Flakes</a>
            <span>/</span>
            <a href="{SITE_URL}/projects/" target="_blank">Projects</a>
            <span>/</span>
            <span>Sentiment analyser</span>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_intro() -> None:
    """Introduce the page."""
    st.title("Text sentiment analyser")
    st.write(
        "Paste some text and see how positive or negative it reads, sentence by "
        "sentence, with two different methods side by side. It also pulls out "
        "the keywords and the people, places and organisations it mentions."
    )


def render_sidebar() -> str:
    """Render the settings.

    Returns:
        The engine used to score individual sentences.
    """
    with st.sidebar:
        st.header("Settings")
        engine = st.segmented_control(
            "Score sentences with",
            ["VADER", "TextBlob"],
            default="VADER",
            help=(
                "VADER suits short, informal text such as reviews and posts. "
                "TextBlob also estimates how subjective the writing is."
            ),
        )
        with st.expander("What do the scores mean?"):
            st.write(
                "Polarity runs from -1 (very negative) to +1 (very positive). "
                "Both methods look up words in a sentiment dictionary, so they "
                "miss sarcasm and context. Treat the result as a guide."
            )
    return str(engine or "VADER")


def render_input() -> str:
    """Show the text box, examples and file upload.

    Returns:
        The text to analyse, trimmed to the length limit.
    """
    st.segmented_control(
        "Try an example", list(EXAMPLES), key="example", on_change=_load_example
    )
    text = st.text_area(
        "Text to analyse",
        key="text",
        height=170,
        placeholder="Paste a review, an email or a paragraph of news...",
    )
    st.file_uploader(
        "Or upload a .txt file", type="txt", key="upload", on_change=_load_upload
    )
    if len(text) > MAX_CHARS:
        st.warning(f"Only the first {MAX_CHARS:,} characters are analysed.")
    return str(text)[:MAX_CHARS]


def render_verdict(text: str) -> None:
    """Show both engines' overall result and whether they agree.

    Args:
        text: The text to analyse.
    """
    result = compare_engines(text)
    vader, blob = result["vader"], result["textblob"]
    cols = st.columns(3)
    cols[0].metric(f"VADER ({vader['polarity']:+.2f})", vader["sentiment"])
    cols[1].metric(f"TextBlob ({blob['polarity']:+.2f})", blob["sentiment"])
    cols[2].metric(
        "Subjectivity",
        f"{blob['subjectivity']:.0%}",
        help="0% is factual, 100% is opinion. TextBlob only.",
    )
    if not result["agree"]:
        st.caption(
            "The two methods disagree on the overall label. That usually means "
            "the text is mixed or close to neutral."
        )
    st.plotly_chart(
        draw_engine_comparison(blob["polarity"], vader["polarity"]), width="stretch"
    )


def highlight_html(scores: list[SentenceScore]) -> str:
    """Render the text with each sentence tinted by its tone.

    All user text is escaped before it is placed in the markup.

    Args:
        scores: Sentence scores in reading order.

    Returns:
        An HTML paragraph.
    """
    parts = []
    for item in scores:
        polarity = item["polarity"]
        safe = html.escape(item["text"])
        if polarity == 0:
            parts.append(f"<span>{safe}</span>")
            continue
        rgb = POSITIVE_RGB if polarity > 0 else NEGATIVE_RGB
        alpha = min(0.12 + 0.3 * abs(polarity), 0.42)
        style = f"background: rgba({rgb},{alpha:.2f})"
        title = html.escape(f"{polarity:+.2f}")
        parts.append(f'<span class="hl" style="{style}" title="{title}">{safe}</span>')
    return '<p class="reading">' + " ".join(parts) + "</p>"


def render_timeline(text: str, engine: str) -> None:
    """Show how the tone moves through the text, and the text itself, tinted.

    Args:
        text: The text to analyse.
        engine: The engine used for each sentence.
    """
    scores = score_sentences(text, engine)
    st.subheader("How the tone moves")
    st.caption(
        f"Each sentence scored with {engine}. Blue reads positive, rose reads "
        "negative; hover a sentence for its score."
    )
    st.plotly_chart(draw_sentence_timeline(scores), width="stretch")
    st.markdown(highlight_html(scores), unsafe_allow_html=True)


def _chips(items: list[str]) -> str:
    return "".join(f'<span class="chip">{html.escape(item)}</span>' for item in items)


def render_details(text: str) -> None:
    """Show keywords, entities and text statistics, one per tab.

    Args:
        text: The text to analyse.
    """
    st.subheader("Details")
    why, keywords, entities, about = st.tabs(
        ["Why this score", "Keywords", "Entities", "About the text"]
    )
    with why:
        words = sentiment_words(text)
        if words:
            st.caption(
                "Words from the text that VADER's dictionary rates as positive "
                "(blue) or negative (rose). The longer the bar, the stronger the pull."
            )
        st.plotly_chart(draw_word_valence(words), width="stretch")
    with keywords:
        counts = get_word_frequency(text)
        if counts and max(counts.values()) > 1:
            st.caption("Most frequent words, ignoring common ones like 'the'.")
            st.plotly_chart(draw_keyword_chart(counts), width="stretch")
        elif counts:
            st.caption("No word repeats, so there is nothing to rank. Key words:")
            st.markdown(_chips(list(counts)), unsafe_allow_html=True)
        else:
            st.info("Not enough text to find keywords.")
    with entities:
        found = get_named_entities(text)
        if not found:
            st.info("No people, places or organisations found in this text.")
        grouped: dict[str, set[str]] = defaultdict(set)
        for entity in found:
            grouped[ENTITY_LABELS.get(entity["label"], entity["label"])].add(
                entity["text"]
            )
        for label in sorted(grouped):
            st.markdown(f"**{label}**")
            st.markdown(_chips(sorted(grouped[label])), unsafe_allow_html=True)
    with about:
        stats = text_stats(text)
        cols = st.columns(4)
        cols[0].metric("Words", f"{stats['words']:,}")
        cols[1].metric("Sentences", f"{stats['sentences']:,}")
        cols[2].metric("Words per sentence", f"{stats['words_per_sentence']:.1f}")
        cols[3].metric(
            "Distinct words",
            f"{stats['unique_share']:.0%}",
            help="Share of words that are not repeats. Lower means more repetition.",
        )


def render_footer() -> None:
    """Show a one-line footer linking back to the Data Flakes site."""
    st.markdown(
        f"""
        <footer class="site-foot">
            Built by Elliott Fairhall. More projects and write-ups at
            <a href="{SITE_URL}/" target="_blank">data-flakes.dev</a>.
        </footer>
        """,
        unsafe_allow_html=True,
    )
