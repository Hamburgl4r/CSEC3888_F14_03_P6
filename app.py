"""Streamlit prototype for the Civil Liability Act research application.

Run from the repository root:

    streamlit run app.py

Prototype mode uses a few sample records while the retrieval modules are being
completed. Turn it off when ``retrieval.retrieve.search`` is ready.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import date
from typing import Any

import streamlit as st


def add_page_styles() -> None:
    """Apply the visual system used by the research interface."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,650;6..72,750&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --ink: #18242a;
            --muted: #5a696e;
            --navy: #0b2735;
            --navy-soft: #163b4a;
            --paper: #f2eee5;
            --surface: #fffdf8;
            --line: #9ba6a8;
            --accent: #db4d2f;
            --accent-dark: #b83821;
            --gold: #e8b949;
            --body: "Space Grotesk", "Avenir Next", "Century Gothic", sans-serif;
            --display: "Newsreader", Georgia, "Times New Roman", serif;
        }

        .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--paper);
            color: var(--ink);
            font-family: var(--body);
        }

        header[data-testid="stHeader"] {
            background: var(--navy);
            border-bottom: 4px solid var(--accent);
        }

        .block-container {
            max-width: 1260px;
            padding: 4rem 3rem 5rem;
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            font-family: var(--display);
        }

        .masthead {
            align-items: center;
            border-bottom: 1px solid var(--navy);
            display: flex;
            justify-content: space-between;
            margin-bottom: 2.35rem;
            padding-bottom: 0.8rem;
        }

        .masthead-name {
            color: var(--navy);
            font-family: var(--display);
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -0.025em;
        }

        .masthead-scope {
            color: var(--muted);
            font-size: 0.69rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .hero-copy {
            animation: reveal-up 420ms ease-out both;
            padding: 0.65rem 2.5rem 1.3rem 0;
        }

        .hero-kicker {
            align-items: center;
            color: var(--accent);
            display: flex;
            font-size: 0.7rem;
            font-weight: 700;
            gap: 0.7rem;
            letter-spacing: 0.13em;
            margin: 0 0 1rem;
            text-transform: uppercase;
        }

        .hero-kicker::before {
            background: var(--accent);
            content: "";
            height: 2px;
            width: 2.75rem;
        }

        .page-title {
            color: var(--navy);
            font-family: var(--display);
            font-size: clamp(3.15rem, 6.8vw, 6.2rem);
            font-weight: 650;
            letter-spacing: -0.06em;
            line-height: 0.88;
            margin: 0;
            max-width: 850px;
        }

        .page-intro {
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.65;
            margin: 1.55rem 0 0 5.7rem;
            max-width: 740px;
        }

        .collection-line {
            align-items: center;
            border-bottom: 1px solid #aab2b4;
            border-top: 1px solid #aab2b4;
            color: var(--navy);
            display: flex;
            flex-wrap: wrap;
            font-size: 0.66rem;
            font-weight: 700;
            gap: 0;
            letter-spacing: 0.1em;
            margin: 1.35rem 0 0 5.7rem;
            max-width: 740px;
            padding: 0.65rem 0;
            text-transform: uppercase;
        }

        .collection-line span + span::before {
            color: var(--accent);
            content: "/";
            padding: 0 0.8rem;
        }

        .trust-strip {
            background: var(--gold);
            color: #2a261b;
            font-size: 0.8rem;
            line-height: 1.5;
            margin: 0 0 1.65rem 5.7rem;
            max-width: 740px;
            padding: 0.8rem 1rem;
        }

        .trust-strip strong {
            color: #17140d;
        }

        .form-label {
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }

        .form-heading {
            color: var(--navy);
            font-family: var(--display);
            font-size: 2rem;
            font-weight: 650;
            letter-spacing: -0.025em;
            line-height: 1.1;
            margin: 0 0 1.2rem;
        }

        .filter-heading {
            border-bottom: 1px solid #96a2a6;
            color: var(--navy);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.85rem;
            padding-bottom: 0.65rem;
            text-transform: uppercase;
        }

        .section-heading {
            align-items: baseline;
            border-bottom: 2px solid var(--navy);
            color: var(--navy);
            display: flex;
            font-family: var(--display);
            font-size: 2rem;
            font-weight: 650;
            justify-content: space-between;
            letter-spacing: -0.025em;
            margin: 4rem 0 0.7rem;
            padding-bottom: 0.7rem;
        }

        .search-summary {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 1.4rem;
        }

        .result-index {
            color: var(--accent);
            font-family: var(--display);
            font-size: 2.5rem;
            font-weight: 650;
            letter-spacing: -0.06em;
            line-height: 1;
            padding-top: 0.2rem;
        }

        .result-type {
            color: var(--accent);
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin: 0 0 0.45rem;
            text-transform: uppercase;
        }

        .result-meta {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.55;
            margin: -0.2rem 0 0.45rem;
        }

        .match-note {
            color: #68787d;
            font-size: 0.72rem;
            margin: 0 0 0.95rem;
        }

        .source-text {
            background: #e9e4da;
            border-left: 5px solid var(--navy-soft);
            color: #17252b;
            font-family: var(--display);
            font-size: 1rem;
            line-height: 1.72;
            padding: 1.2rem 1.35rem;
        }

        .passage-label {
            border-bottom: 1px solid #98a4a8;
            color: var(--navy);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 1rem 0 0;
            padding: 0 0 0.55rem;
            text-transform: uppercase;
        }

        div[data-testid="stForm"] {
            background: var(--surface);
            border: 2px solid var(--navy);
            border-radius: 0;
            box-shadow: 9px 9px 0 var(--navy);
            margin: 0 9px 3rem 0;
            padding: 1.6rem 1.7rem 1rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent;
            border: 0 !important;
            border-bottom: 1px solid #aab2b4 !important;
            border-radius: 0;
            border-top: 2px solid var(--navy) !important;
            padding: 1.45rem 0 1.6rem;
            transition: border-color 150ms ease, transform 150ms ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-top-color: var(--accent) !important;
            transform: translateX(5px);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] h3 {
            color: var(--navy) !important;
            font-family: var(--display);
            font-size: 1.55rem;
            font-weight: 650;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: #fffefb !important;
            border-color: #7f8e93 !important;
            border-radius: 0 !important;
            color: var(--ink) !important;
            font-family: var(--body) !important;
        }

        [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stTextInput"] input::placeholder {
            color: #65757d !important;
            opacity: 1 !important;
        }

        [data-testid="stFormSubmitButton"] button {
            background: var(--accent) !important;
            border: 2px solid var(--accent) !important;
            border-radius: 0 !important;
            color: #ffffff !important;
            font-family: var(--body) !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em;
            min-height: 2.9rem;
            transition: background 130ms ease, transform 130ms ease;
        }

        [data-testid="stFormSubmitButton"] button:hover {
            background: var(--accent-dark) !important;
            border-color: var(--accent-dark) !important;
            transform: translate(-3px, -3px);
        }

        [data-testid="stLinkButton"] a {
            background: transparent;
            border: 1px solid var(--navy);
            border-radius: 0;
            color: var(--navy) !important;
            font-family: var(--body);
            transition: background 130ms ease, color 130ms ease;
        }

        [data-testid="stLinkButton"] a:hover {
            background: var(--navy);
            color: #ffffff !important;
        }

        [data-testid="stExpander"] {
            background: transparent;
            border: 1px solid #98a4a8 !important;
            border-radius: 0 !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: var(--navy) !important;
            font-family: var(--body);
            font-weight: 600;
        }

        [data-testid="stRadio"] label {
            border: 1px solid #879398;
            margin-right: -1px;
            padding: 0.4rem 0.8rem;
            transition: background 120ms ease, color 120ms ease;
        }

        [data-testid="stRadio"] label:has(input:checked) {
            background: var(--navy);
            color: #ffffff !important;
        }

        [data-testid="stRadio"] label:has(input:checked) * {
            color: #ffffff !important;
        }

        [data-testid="stAlert"] {
            color: var(--ink);
        }

        .search-guide {
            border-bottom: 1px solid var(--navy);
            border-top: 2px solid var(--navy);
            display: grid;
            grid-template-columns: 1.2fr 0.8fr 1fr;
            margin: 4.5rem 0 0;
        }

        .guide-item {
            min-height: 165px;
            padding: 1.25rem 1.5rem 1.5rem 0;
        }

        .guide-item + .guide-item {
            border-left: 1px solid #aab2b4;
            padding-left: 1.5rem;
        }

        .guide-number {
            color: var(--accent);
            font-family: var(--display);
            font-size: 1.8rem;
            font-weight: 650;
        }

        .guide-title {
            color: var(--navy);
            font-family: var(--display);
            font-size: 1.2rem;
            font-weight: 650;
            margin: 0.5rem 0;
        }

        .guide-copy {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.55;
        }

        .footer-note {
            border-top: 1px solid var(--line);
            color: #647277;
            font-size: 0.72rem;
            line-height: 1.5;
            margin-top: 3rem;
            padding-top: 1rem;
        }

        @keyframes reveal-up {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation: none !important;
                scroll-behavior: auto !important;
                transition: none !important;
            }
        }

        @media (max-width: 700px) {
            .block-container {
                padding: 4.8rem 1.25rem 3rem;
            }
            .masthead {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.25rem;
            }
            .page-title {
                font-size: 3.15rem;
            }
            .page-intro,
            .collection-line,
            .trust-strip {
                margin-left: 0;
            }
            .search-guide {
                display: block;
            }
            .guide-item + .guide-item {
                border-left: 0;
                border-top: 1px solid #aab2b4;
                padding-left: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def paragraph_label(numbers: list[int]) -> str:
    """Return a short citation label for one or more paragraph numbers."""
    if not numbers:
        return "Paragraph unavailable"
    if len(numbers) == 1:
        return f"[{numbers[0]}]"
    return f"[{numbers[0]}]-[{numbers[-1]}]"


def render_source_text(text: str) -> None:
    """Display source text with safe HTML and readable legal typography."""
    escaped = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f'<div class="source-text">{escaped}</div>',
        unsafe_allow_html=True,
    )


def render_labeled_source(label: str, text: str) -> None:
    """Display a citation label and its passage without layered controls."""
    st.markdown(
        f'<p class="passage-label">{html.escape(label)}</p>',
        unsafe_allow_html=True,
    )
    render_source_text(text)


def run_search(
    query: str,
    *,
    court: str | None,
    date_from: date,
    date_to: date,
    provision: str | None,
    search_function: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Pass the form values to the shared hybrid retrieval function."""
    if search_function is None:
        from retrieval.retrieve import search

        search_function = search

    return search_function(
        query,
        court=court,
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        provision=provision,
    )


def render_legislation(item: dict[str, Any], position: int) -> None:
    """Display one legislation result and its exact provision text."""
    with st.container(border=True):
        marker, content = st.columns([0.1, 0.9], gap="medium")
        with marker:
            st.markdown(
                f'<div class="result-index">{position:02}</div>',
                unsafe_allow_html=True,
            )
        with content:
            st.markdown(
                '<p class="result-type">Legislation</p>',
                unsafe_allow_html=True,
            )
            st.subheader(f"{item['provision']} {item.get('heading', '')}")
            st.markdown(
                f"<p class='result-meta'>{item.get('part_heading', '')} "
                f"· {item.get('division_heading', '')}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='match-note'>Hybrid retrieval result</p>",
                unsafe_allow_html=True,
            )
            render_labeled_source(
                f"Cited provision · {item['provision']}",
                item["text"],
            )
            st.link_button("Official source", item["url"])


def render_judgment(item: dict[str, Any], position: int) -> None:
    """Display one judgment result with a safe paragraph citation."""
    with st.container(border=True):
        marker, content = st.columns([0.1, 0.9], gap="medium")
        with marker:
            st.markdown(
                f'<div class="result-index">{position:02}</div>',
                unsafe_allow_html=True,
            )
        with content:
            st.markdown(
                '<p class="result-type">Judgment</p>',
                unsafe_allow_html=True,
            )
            st.subheader(item["citation"])
            st.markdown(
                f"<p class='result-meta'>{item.get('court', '')} "
                f"· {item.get('date', '')} · {item.get('catchwords', '')}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='match-note'>Hybrid retrieval result</p>",
                unsafe_allow_html=True,
            )

            if item.get("citation_available"):
                citation = paragraph_label(item.get("paragraph_numbers", []))
                render_labeled_source(
                    f"Cited passage · {citation}",
                    item["text"],
                )
            else:
                st.warning(
                    "This passage can be searched, but a reliable paragraph "
                    "number is not available."
                )
                render_labeled_source(
                    "Passage without pinpoint citation",
                    item["text"],
                )

            st.link_button("Official source", item["url"])


def render_results(results: list[dict[str, Any]]) -> None:
    """Separate legislation and judgment results as required by the brief."""
    legislation = [
        item for item in results if item.get("document_type") == "legislation"
    ]
    judgments = [
        item for item in results if item.get("document_type") == "judgment"
    ]

    view = st.radio(
        "Result type",
        [
            f"All {len(results)}",
            f"Legislation {len(legislation)}",
            f"Judgments {len(judgments)}",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view.startswith("Legislation"):
        visible_results = legislation
    elif view.startswith("Judgments"):
        visible_results = judgments
    else:
        visible_results = results

    if not visible_results:
        st.info(
            f"No {view.split()[0].lower()} passages were returned for this "
            "search. Choose another result type or run a broader search."
        )
        return

    for position, item in enumerate(visible_results, start=1):
        if item.get("document_type") == "legislation":
            render_legislation(item, position)
        else:
            render_judgment(item, position)

def main() -> None:
    st.set_page_config(
        page_title="CLA Research",
        page_icon="§",
        layout="wide",
    )
    add_page_styles()

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    st.markdown(
        """
        <div class="masthead">
            <span class="masthead-name">CLA / NSW</span>
            <span class="masthead-scope">Research index · prototype 01</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-copy">
            <p class="hero-kicker">NSW legal research</p>
            <h1 class="page-title">Find the paragraph that matters.</h1>
            <p class="page-intro">
                Search the <em>Civil Liability Act 2002</em> and the NSW
                judgments that apply it. Start with facts, a section, or the
                name of a case.
            </p>
            <p class="collection-line">
                <span>2,386 judgments</span>
                <span>1 Act</span>
                <span>3 NSW courts</span>
                <span>2010 onward</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="trust-strip">
            <strong>Research material, not legal advice.</strong>
            This service reports what published documents say and does not
            predict outcomes. Check every result against the official source.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("search_form"):
        question_column, filter_column = st.columns(
            [1.7, 0.85],
            gap="large",
            vertical_alignment="top",
        )
        with question_column:
            st.markdown(
                """
                <p class="form-label">Research query</p>
                <p class="form-heading">What happened?</p>
                """,
                unsafe_allow_html=True,
            )
            query = st.text_area(
                "Research question",
                placeholder=(
                    "Example: A visitor slipped on a wet supermarket floor. "
                    "Which provisions and cases discuss reasonable precautions "
                    "and causation?"
                ),
                height=178,
                label_visibility="collapsed",
            )
            st.caption(
                "Plain language works. Exact case names and section numbers "
                "work too."
            )

        with filter_column:
            st.markdown(
                '<p class="filter-heading">Limit the search</p>',
                unsafe_allow_html=True,
            )
            court_choice = st.selectbox(
                "Court",
                ["All courts", "NSWSC", "NSWCA", "NSWDC"],
            )
            provision = st.text_input(
                "Provision",
                placeholder="For example, 5D",
            )
            date_columns = st.columns(2, gap="small")
            with date_columns[0]:
                date_from = st.date_input(
                    "From",
                    value=date(2010, 1, 1),
                    min_value=date(2010, 1, 1),
                )
            with date_columns[1]:
                date_to = st.date_input("To", value=date.today())

        action_copy, action_button = st.columns(
            [1.7, 0.85],
            gap="large",
            vertical_alignment="bottom",
        )
        with action_copy:
            st.caption(
                "Hybrid search combines exact keyword matching with semantic "
                "similarity."
            )
        with action_button:
            submitted = st.form_submit_button(
                "Run search",
                type="primary",
                use_container_width=True,
            )

    if not submitted and st.session_state.search_results is None:
        st.markdown(
            """
            <section class="search-guide">
                <div class="guide-item">
                    <div class="guide-number">01</div>
                    <div class="guide-title">Begin with the facts</div>
                    <div class="guide-copy">
                        Describe the situation as you understand it. Legal
                        terminology is optional.
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-number">02</div>
                    <div class="guide-title">Narrow the record</div>
                    <div class="guide-copy">
                        Filter by court, date, or an exact provision such as
                        5B or 5D.
                    </div>
                </div>
                <div class="guide-item">
                    <div class="guide-number">03</div>
                    <div class="guide-title">Read the authority</div>
                    <div class="guide-copy">
                        Inspect the cited paragraph and open the official
                        source before relying on a result.
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    if submitted:
        st.session_state.search_results = None
        st.session_state.search_query = query.strip()

        if not query.strip():
            st.error("Enter a situation or legal search term.")
            return

        if date_from > date_to:
            st.error("The start date must be earlier than the end date.")
            return

        try:
            with st.spinner("Searching the knowledge base..."):
                results = run_search(
                    query,
                    court=None if court_choice == "All courts" else court_choice,
                    date_from=date_from,
                    date_to=date_to,
                    provision=provision or None,
                )
        except FileNotFoundError:
            st.error(
                "The local search index was not found. Run "
                "`python scripts/build_indexes.py` from the project root, "
                "then try again."
            )
            return
        except ValueError as error:
            st.error(f"The search could not be completed: {error}")
            return
        except Exception as error:
            st.error(
                "The search service could not start. Check that the processed "
                "data, model, and saved index are available."
            )
            with st.expander("Technical details"):
                st.code(str(error))
            return

        st.session_state.search_results = results

    results = st.session_state.search_results
    if results is None:
        return

    if not results:
        st.info(
            "No matching material was found. Try removing a filter or using "
            "different words."
        )
        return

    st.markdown(
        '<h2 class="section-heading">Search results</h2>',
        unsafe_allow_html=True,
    )
    safe_query = html.escape(st.session_state.search_query)
    st.markdown(
        f'<p class="search-summary">{len(results)} passages found for '
        f'<strong>{safe_query}</strong>. Results are ordered by relevance.</p>',
        unsafe_allow_html=True,
    )
    render_results(results)

    st.markdown(
        """
        <p class="footer-note">
            Source collection: Open Australian Legal Corpus. Reproduced
            material is provided for research and must be checked against the
            official source.
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
