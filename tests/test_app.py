from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import run_search


def test_run_search_passes_frontend_filters_to_retrieval():
    calls = []

    def fake_search(query, **filters):
        calls.append((query, filters))
        return [{"chunk_id": "result_1"}]

    results = run_search(
        "slip on a wet floor",
        court="NSWCA",
        date_from=date(2012, 1, 1),
        date_to=date(2024, 12, 31),
        provision="s 5D",
        search_function=fake_search,
    )

    assert results == [{"chunk_id": "result_1"}]
    assert calls == [
        (
            "slip on a wet floor",
            {
                "court": "NSWCA",
                "date_from": "2012-01-01",
                "date_to": "2024-12-31",
                "provision": "s 5D",
            },
        )
    ]


def test_result_type_change_keeps_stored_results():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    page = AppTest.from_file(app_path)
    page.session_state["search_query"] = "duty of care"
    page.session_state["search_results"] = [
        {
            "chunk_id": "case_1",
            "document_type": "judgment",
            "citation": "Example v Council [2020] NSWCA 1",
            "court": "NSWCA",
            "date": "2020-01-10",
            "catchwords": "Negligence; duty of care",
            "paragraph_numbers": [12],
            "citation_available": True,
            "text": "[12] The court considered the duty of care.",
            "url": "https://example.test/case",
        }
    ]

    page.run(timeout=20)
    page.radio[0].set_value("Legislation 0").run(timeout=20)
    assert "No legislation passages" in page.info[0].value

    page.radio[0].set_value("Judgments 1").run(timeout=20)

    assert not page.exception
    assert page.session_state["search_query"] == "duty of care"
    assert "Example v Council [2020] NSWCA 1" in page.subheader.values
