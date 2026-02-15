"""UI text content and documentation for the sentiment analysis dashboard.

This module provides functions that render explanatory text and documentation
throughout the Streamlit application, keeping content separate from presentation logic.
"""

import streamlit as st

# Header content
HEADER_HTML = """
<div class='header-container'>
    <h1 class='main-title'>Sentiment Dynamics</h1>
    <p class='sub-title'>UNVEILING EMOTION THROUGH DATA</p>
</div>
"""

# Project overview content
PROJECT_OVERVIEW = """
### Project Intelligence

This application modernizes sentiment analysis by providing a multi-dimensional
view of textual data. By leveraging **Natural Language Processing (NLP)** and
pre-trained machine learning models, we transform raw text into actionable insights.

**Key Analytical Pillars:**
- **Sentiment Quantification**: Real-time polarity and subjectivity scoring.
- **Keyword Intelligence**: Automated extraction of thematic anchors.
- **Entity Recognition**: Deep classification of people, organisations, and locations.
"""

# Analysis guide content
ANALYSIS_GUIDE = """
### Analysis Guide

To begin, enter your text in the workspace below and trigger the
**'Process Intelligence'** engine.

- **Polarity**: Measures the emotional lean (-1 to 1).
- **Subjectivity**: Quantifies factual vs. opinionated content (0 to 1).
- **NER**: Identifies the 'who, where, and what' within your text.
"""

# Sentiment definitions
SENTIMENT_DEFINITIONS = """
- **Positive**: Signals happiness, admiration, or alignment.
- **Negative**: Highlights friction, frustration, or concern.
- **Neutral**: Objective, factual, or emotionally balanced reporting.
"""

# Polarity explanation
POLARITY_EXPLANATION = """
Polarity is a core NLP metric. A score of **1** is purely positive,
while **-1** is deeply negative. Neutral statements reside near **0**.
"""

# Subjectivity explanation
SUBJECTIVITY_EXPLANATION = """
A score of **0** indicates pure objectivity (facts), while **1** represents
pure subjectivity (opinions).
"""

# Keyword explanation
KEYWORD_EXPLANATION = """
Keyword extraction reveals the thematic distribution of your text.
The most frequent anchors are displayed to help identify core topics quickly.
"""

# NER explanation
NER_EXPLANATION = """
**Named Entity Recognition (NER)** classifies text into predefined categories like:
- `PERSON`: Specific individuals.
- `ORG`: Companies or institutions.
- `GPE`: Geographical locations (countries, cities).
- `DATE`: Specific time references.
"""


def header_info() -> None:
    """Render the premium header with gradient text.

    Displays the main title and subtitle using custom HTML/CSS styling
    for a premium glassmorphism aesthetic.

    Example:
        >>> header_info()  # Renders header in Streamlit app
    """
    st.markdown(HEADER_HTML, unsafe_allow_html=True)


def project_overview() -> None:
    """Render the project overview section.

    Displays information about the application's capabilities and
    the key analytical features available.

    Example:
        >>> project_overview()  # Renders overview in Streamlit app
    """
    st.markdown(PROJECT_OVERVIEW)


def analysis_guide() -> None:
    """Render the analysis guide in the sidebar.

    Provides users with instructions on how to use the sentiment
    analysis tool and explanations of the metrics.

    Example:
        >>> analysis_guide()  # Renders guide in Streamlit app
    """
    st.markdown(ANALYSIS_GUIDE)


def sentiment_definitions() -> None:
    """Render sentiment classification definitions.

    Explains what positive, negative, and neutral sentiments mean
    in the context of the analysis.

    Example:
        >>> sentiment_definitions()  # Renders definitions in Streamlit app
    """
    st.markdown(SENTIMENT_DEFINITIONS)


def polarity_explained() -> None:
    """Render polarity metric explanation.

    Explains the polarity score range and interpretation.

    Example:
        >>> polarity_explained()  # Renders explanation in Streamlit app
    """
    st.markdown(POLARITY_EXPLANATION)


def subjectivity_explained() -> None:
    """Render subjectivity metric explanation.

    Explains the subjectivity score range and interpretation.

    Example:
        >>> subjectivity_explained()  # Renders explanation in Streamlit app
    """
    st.markdown(SUBJECTIVITY_EXPLANATION)


def keyword_explained() -> None:
    """Render keyword extraction explanation.

    Explains how keyword extraction works and its purpose.

    Example:
        >>> keyword_explained()  # Renders explanation in Streamlit app
    """
    st.markdown(KEYWORD_EXPLANATION)


def ner_explained() -> None:
    """Render named entity recognition explanation.

    Explains the NER feature and the types of entities detected.

    Example:
        >>> ner_explained()  # Renders explanation in Streamlit app
    """
    st.markdown(NER_EXPLANATION)
