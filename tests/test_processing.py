import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from parse_judgements import parse_judgment, parse_paragraphs
from process_corpus import (
    MAX_CHUNK_CHARS,
    extract_chunk_act_references,
    extract_section_references,
    make_judgment_chunks,
    split_legislation_into_sections,
)


class ParagraphParsingTests(unittest.TestCase):
    def test_citation_year_is_not_a_paragraph(self):
        text = """Medium Neutral Citation: Example [2012] NSWSC 1
JUDGMENT
1. First paragraph.
2. Second paragraph cites Example [2020] HCA 2.
3. Third paragraph."""
        parsed = parse_paragraphs(text)
        self.assertEqual(
            [item["paragraph_number"] for item in parsed["paragraphs"]],
            [1, 2, 3],
        )
        self.assertNotIn(2012, [
            item["paragraph_number"] for item in parsed["paragraphs"]
        ])

    def test_reconstructs_top_level_restarts_and_ignores_nested_lists(self):
        text = """JUDGMENT
1. First paragraph.
2. Second paragraph:
        1. nested item
        2. nested item
Heading
1. Third source paragraph after a lost HTML start value.
2. Fourth source paragraph.
3. Fifth source paragraph."""
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "reconstructed")
        self.assertEqual(
            [item["paragraph_number"] for item in parsed["paragraphs"]],
            [1, 2, 3, 4, 5],
        )

    def test_parses_legacy_indented_paragraphs(self):
        text = """  JUDGMENT
  1 First paragraph.
          1 Nested item.
          2 Nested item.
  2 Second paragraph.
  3 Third paragraph."""
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "original")
        self.assertEqual(len(parsed["paragraphs"]), 3)

    def test_marks_unreliable_text_unavailable(self):
        parsed = parse_paragraphs("No numbered paragraphs are present.")
        self.assertEqual(parsed["status"], "unavailable")
        self.assertEqual(parsed["paragraphs"], [])

    def test_parses_glued_markers_without_a_separator(self):
        text = "JUDGMENT\n" + "\n".join(
            f"{number}The court considered the evidence carefully."
            for number in range(1, 26)
        )
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["marker_style"], "glued")
        self.assertEqual(len(parsed["paragraphs"]), 25)
        self.assertEqual(parsed["paragraphs"][0]["text"], "[1] The court considered the evidence carefully.")

    def test_ignores_trailing_orders_list_in_unnumbered_judgment(self):
        # The body carries no markers, and a short numbered list of orders sits
        # at the very end. Labelling that list as the whole judgment would drop
        # the body and produce wrong pinpoint citations.
        body = "The plaintiff gave evidence about the accident. " * 400
        text = f"""Medium Neutral Citation: Example v Person [2020] NSWSC 1
JUDGMENT
{body}
Orders
1. Judgment for the plaintiff.
2. The defendant is to pay costs.
3. Exhibits may be returned."""
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "unavailable")
        self.assertEqual(parsed["paragraphs"], [])

    def test_rejects_markers_when_one_paragraph_swallows_the_body(self):
        # A short orders list inside the header, followed by an unnumbered body.
        text = """Medium Neutral Citation: Example v Person [2020] NSWDC 4
Decision:                 2. Judgment for the defendant.
                          3. The exhibits may be returned.
                          4. Liberty to apply on 7 days notice.
""" + "The court reviewed the medical evidence in detail. " * 800
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "unavailable")
        self.assertEqual(parsed["paragraphs"], [])

    def test_keeps_long_leading_run_when_numbering_restarts_late(self):
        lines = [f"{number}. Substantive paragraph text here." for number in range(10, 60)]
        lines += ["1. A short appended list item.", "2. Another appended item."]
        parsed = parse_paragraphs("JUDGMENT\n" + "\n".join(lines))
        self.assertEqual(parsed["paragraphs"][0]["displayed_number"], 10)
        self.assertGreaterEqual(len(parsed["paragraphs"]), 50)

    def test_long_paragraphs_are_fragmented_with_same_pinpoint(self):
        text = "JUDGMENT\n" + "\n".join(
            [
                "1. " + ("First sentence. " * 500),
                "2. Short second paragraph.",
                "3. Short third paragraph.",
            ]
        )
        doc = {"text": text}
        chunks = make_judgment_chunks(doc)
        self.assertTrue(all(len(chunk["text"]) <= MAX_CHUNK_CHARS for chunk in chunks))
        first_fragments = [
            chunk for chunk in chunks if chunk["paragraph_start"] == 1
        ]
        self.assertGreater(len(first_fragments), 1)
        self.assertTrue(
            all(chunk["paragraph_end"] == 1 for chunk in first_fragments)
        )


class HeaderParsingTests(unittest.TestCase):
    def test_extracts_modern_header_lists(self):
        text = """Medium Neutral Citation: Example v Person [2020] NSWSC 1
Catchwords: NEGLIGENCE — causation
Legislation Cited: Civil Liability Act 2002 (NSW), s 5D(1)(a)
Cases Cited: Strong v Woolworths Ltd [2012] HCA 5
JUDGMENT
1. First.
2. Second.
3. Third."""
        header = parse_judgment(text)["header"]
        self.assertEqual(header["catchwords"], "NEGLIGENCE — causation")
        self.assertEqual(len(header["legislation_cited"]), 1)
        self.assertEqual(len(header["cases_cited"]), 1)

    def test_repairs_legacy_first_case_layout(self):
        text = """  LEGISLATION CITED: Civil Liability Act 2002
                       Brodie v Singleton Shire Council (2001) 206 CLR 512
  CASES CITED:         Wyong v Shirt (1980) 146 CLR 40
  JUDGMENT
  1 First.
  2 Second.
  3 Third."""
        header = parse_judgment(text)["header"]
        self.assertEqual(header["legislation_cited"], ["Civil Liability Act 2002"])
        self.assertEqual(len(header["cases_cited"]), 2)


class LegislationParsingTests(unittest.TestCase):
    def test_main_sections_and_schedule_clauses_have_unique_ids(self):
        text = """Civil Liability Act 2002 No 22
Part 1A Negligence
Division 3 Causation
5D General principles
    (1) Some text.
    2002 amending Act means another Act.
Schedule 1 Savings and transitional provisions
Part 1 Preliminary
1 Regulations
    Schedule text.
Schedule 2 Transferred provisions
1 Abolition of action
    More text."""
        provisions = split_legislation_into_sections(text)
        self.assertEqual(
            [item["provision_id"] for item in provisions],
            ["s_5D", "sch_1_cl_1", "sch_2_cl_1"],
        )
        self.assertEqual(len({item["provision_id"] for item in provisions}), 3)
        self.assertEqual(provisions[0]["part"], "1A")
        self.assertEqual(provisions[0]["division"], "3")


class ActReferenceTests(unittest.TestCase):
    def setUp(self):
        self.order = ["1", "5B", "5C", "5D", "5E", "56"]
        self.valid = set(self.order)

    def test_subsection_numbers_do_not_become_sections(self):
        refs = extract_section_references(
            "Civil Liability Act 2002, s 5D(1)(a)", self.valid, self.order
        )
        self.assertEqual(refs, ["5D"])

    def test_accepts_unspaced_section_reference(self):
        refs = extract_section_references(
            "Civil Liability Act 2002 - s5D and s5E.", self.valid, self.order
        )
        self.assertEqual(refs, ["5D", "5E"])

    def test_expands_section_ranges(self):
        refs = extract_section_references(
            "ss 5B–5D", self.valid, self.order
        )
        self.assertEqual(refs, ["5B", "5C", "5D"])

    def test_rejects_foreign_act_shorthand(self):
        sections, _ = extract_chunk_act_references(
            "Section 56 of the Civil Procedure Act applies. "
            "The court also considered s 5D.",
            ["5D"],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, ["5D"])

    def test_accepts_explicit_civil_liability_act_reference(self):
        sections, _ = extract_chunk_act_references(
            "Section 56 of the Civil Liability Act 2002 applies.",
            [],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, ["56"])


if __name__ == "__main__":
    unittest.main()
