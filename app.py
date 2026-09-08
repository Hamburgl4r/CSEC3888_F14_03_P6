"""Streamlit page for the September retrieval demo.

The 14 September checkpoint needs search over the real corpus and a click
through to the correct paragraph. Generated answers are not required yet.

Run from the repository root once the retrieval functions are implemented:

    streamlit run app.py
"""

from __future__ import annotations


def main() -> None:
    import streamlit as st

    from retrieval.retrieve import search

    st.set_page_config(page_title="CLA research demo", layout="wide")
    st.title("Civil Liability Act research")
    st.caption(
        "This tool reports what the documents say. It is not legal advice."
    )

    query = st.text_input("Describe the situation")
    court = st.selectbox("Court", ["All", "NSWSC", "NSWCA", "NSWDC"])
    date_from = st.text_input("Date from (YYYY-MM-DD)", value="2010-01-01")
    date_to = st.text_input("Date to (YYYY-MM-DD)")
    provision = st.text_input("Provision (for example 5D)")

    if not st.button("Search") or not query.strip():
        return

    results = search(
        query,
        court=None if court == "All" else court,
        date_from=date_from or None,
        date_to=date_to or None,
        provision=provision or None,
    )

    for item in results:
        title = item.get("citation") or item.get("provision") or item.get("chunk_id")
        st.subheader(title)
        if item.get("citation_available"):
            st.write("Paragraphs", item.get("paragraph_numbers"))
        else:
            st.write("No pinpoint paragraph citation for this passage.")
        st.write(item.get("text", ""))
        if item.get("url"):
            st.markdown(f"[Source]({item['url']})")


if __name__ == "__main__":
    main()
