"""Sentiment Dynamics - Interactive Sentiment Analysis Dashboard.

A Streamlit application for analyzing text sentiment using TextBlob and VADER,
with keyword extraction and named entity recognition capabilities.
"""

import logging
from collections import defaultdict
from pathlib import Path

import streamlit as st

from utils import analytics, plots, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title="Sentiment Dynamics",
    page_icon=":brain:",
    layout="wide",
)

# Paths
CURRENT_DIR = Path(__file__).parent
CSS_FILE = CURRENT_DIR / "styles" / "main.css"
ASSETS_DIR = CURRENT_DIR / "assets" / "images"
SCENTIMENT_IMAGE = ASSETS_DIR / "Scentiment.jpg"


def load_css(file_path: Path) -> None:
    """Load and inject custom CSS into the Streamlit app.

    Args:
        file_path: Path to the CSS file to load.
    """
    if file_path.exists():
        with open(file_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning("CSS file not found: %s", file_path)


def render_sidebar() -> str:
    """Render the sidebar configuration panel.

    Returns:
        Selected analyzer type ("TextBlob" or "VADER").
    """
    with st.sidebar:
        st.markdown(
            "<h2 style='font-family:Outfit; margin-bottom:0;'>Data Engine</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Configuring high-fidelity signals.")
        st.markdown("---")

        analyzer_type: str = st.radio(
            "Sentiment Analyzer",
            options=["TextBlob", "VADER"],
            help="TextBlob is better for subjectivity. VADER is tuned for social media.",
        )

        st.markdown("---")
        st.caption("Environment: Sentiment Dynamics V4.0")
        st.caption("Aesthetic: Rose / Sky / Glass")

        with st.expander("Quick Start Guide"):
            text.analysis_guide()

    return analyzer_type


def render_overview_tab(
    sentiment: str,
    polarity: float,
    subjectivity: float | None,
) -> None:
    """Render the Overview tab with key metrics.

    Args:
        sentiment: Detected sentiment label (Positive, Neutral, Negative).
        polarity: Polarity score from -1 to 1.
        subjectivity: Subjectivity score from 0 to 1, or None if not available.
    """
    st.markdown("### Analysis Snapshot")

    c1, c2, c3 = st.columns(3)
    c1.metric("Detected Sentiment", sentiment)
    c2.metric("Polarity Score", f"{polarity:.2f}")

    if subjectivity is not None:
        c3.metric("Subjectivity Index", f"{subjectivity:.2f}")
    else:
        c3.metric("Subjectivity Index", "N/A")

    st.markdown("---")

    if SCENTIMENT_IMAGE.exists():
        st.image(str(SCENTIMENT_IMAGE), width="stretch")


def render_nlp_metrics_tab(polarity: float, subjectivity: float | None) -> None:
    """Render the NLP Metrics tab with gauge and pie charts.

    Args:
        polarity: Polarity score from -1 to 1.
        subjectivity: Subjectivity score from 0 to 1, or None if not available.
    """
    st.markdown("### Deep Sentiment Metrics")

    col_l, col_r = st.columns(2)

    with col_l:
        st.plotly_chart(
            plots.draw_polarity_gauge(polarity),
            width="stretch",
        )
        text.polarity_explained()

    with col_r:
        if subjectivity is not None:
            fig = plots.draw_subjectivity_pie(subjectivity)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            text.subjectivity_explained()
        else:
            st.info("Subjectivity analysis is not supported by the VADER engine.")

    st.markdown("---")
    text.sentiment_definitions()


def render_keywords_tab(word_counts: dict[str, int]) -> None:
    """Render the Keywords tab with frequency chart.

    Args:
        word_counts: Dictionary mapping words to frequency counts.
    """
    st.markdown("### Keyword Intelligence")

    if word_counts:
        st.plotly_chart(
            plots.draw_keyword_chart(word_counts),
            width="stretch",
        )
        text.keyword_explained()
    else:
        st.warning("Insufficient vocabulary for keyword extraction.")


def render_entities_tab(entities: list[dict[str, str]]) -> None:
    """Render the Entities tab with NER results.

    Args:
        entities: List of entity dictionaries with 'text' and 'label' keys.
    """
    st.markdown("### Entity Recognition")

    if entities:
        st.markdown("#### Identified Entities")

        # Group entities by label
        grouped: dict[str, list[str]] = defaultdict(list)
        for ent in entities:
            grouped[ent["label"]].append(ent["text"])

        for label, texts in grouped.items():
            st.markdown(f"**{label}**")
            unique_texts = sorted(set(texts))
            tags_html = "".join(
                [f"<span class='ner-tag ner-{label}'>{t}</span>" for t in unique_texts]
            )
            st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.info("No significant entities identified in the provided text.")

    st.markdown("---")
    text.ner_explained()


def main() -> None:
    """Main application entry point."""
    # Load CSS
    load_css(CSS_FILE)

    # Premium Header
    text.header_info()

    # Sidebar Configuration
    analyzer_type = render_sidebar()

    # Main Input Section
    st.markdown("### Input Workspace")

    with st.form(key="nlpForm"):
        raw_text = st.text_area(
            "Enter text for analysis below:",
            height=150,
            placeholder="Paste your text here...",
        )
        submit_button = st.form_submit_button(label="Process Intelligence")

    if submit_button:
        if not raw_text or len(raw_text.strip()) < 5:
            st.warning("Please provide more text for analysis (minimum 5 characters).")
            return

        logger.info("Processing text with %s analyzer", analyzer_type)

        # Perform Analysis
        if analyzer_type == "TextBlob":
            results = analytics.analyze_sentiment_textblob(raw_text)
        else:
            results = analytics.analyze_sentiment_vader(raw_text)

        word_counts = analytics.get_word_frequency(raw_text)
        entities = analytics.get_named_entities(raw_text)

        # Tabs for Results
        t1, t2, t3, t4, t5 = st.tabs(
            [
                "Project Overview",
                "Analysis Snapshot",
                "NLP Metrics",
                "Keywords",
                "Entities",
            ]
        )

        with t1:
            text.project_overview()
            if SCENTIMENT_IMAGE.exists():
                st.image(str(SCENTIMENT_IMAGE), width="stretch")

        with t2:
            render_overview_tab(
                results["sentiment"],
                results["polarity"],
                results.get("subjectivity"),
            )

        with t3:
            render_nlp_metrics_tab(
                results["polarity"],
                results.get("subjectivity"),
            )

        with t4:
            render_keywords_tab(word_counts)

        with t5:
            render_entities_tab(entities)

    else:
        # Initial Landing State
        st.info(
            "Enter text above and click 'Process Intelligence' to generate insights."
        )
        text.project_overview()


if __name__ == "__main__":
    main()
