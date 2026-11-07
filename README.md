# Text sentiment analyser

> Architecture and design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://elliottfairhall-interactive-sentiment-analysis-main-62u54w.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Paste text or upload a .txt file and see how positive or negative it reads, sentence by sentence, with two methods side
by side. Built with Streamlit, TextBlob, VADER, spaCy and Plotly.

## What it does

- VADER and TextBlob results side by side, with a note when they disagree
- A bar for every sentence showing how the tone moves, and the text itself tinted by tone
- A 'Why this score' view of the words driving the result
- Keywords, named entities (people, places, organisations, dates) and text statistics
- Example texts to try; input is limited to 20,000 characters

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs Python 3.11 or newer. The theme follows your system's light or dark setting and is defined in
`.streamlit/config.toml`.

## Test it

```bash
pip install -e ".[dev]"
pytest
```

The tests cover the analysis code, the page itself (run headlessly with Streamlit's `AppTest`) and the dependency pins:
`tests/test_pillow_security_*.py` fails if Pillow is capped below 13 or Python 3.10 is allowed, because older Pillow
releases carry known security advisories.

## Notes

Both methods score words from a dictionary, so they miss sarcasm and context. The spaCy English model is installed from
`requirements.txt`; it loads on first use rather than when the app starts.

Part of [Data Flakes](https://www.data-flakes.dev/).
