"""Text sentiment analyser: entry point."""

import streamlit as st

from utils.layout import (
    MIN_CHARS,
    render_config,
    render_details,
    render_footer,
    render_input,
    render_intro,
    render_sidebar,
    render_timeline,
    render_topbar,
    render_verdict,
)


def main() -> None:
    """Render the page. The analysis updates whenever the text changes."""
    render_config()
    render_topbar()
    render_intro()
    engine = render_sidebar()
    text = render_input()

    if len(text.strip()) < MIN_CHARS:
        st.info("Paste at least a sentence, or pick an example above, to begin.")
        render_footer()
        return

    st.subheader("Overall tone")
    render_verdict(text)
    render_timeline(text, engine)
    render_details(text)
    render_footer()


if __name__ == "__main__":
    main()
