import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from parse_judgements import parse_judgment, parse_paragraphs
from process_corpus import (
    MAX_CHUNK_CHARS,
    build_baseline_case_index,
    build_case_references,
    build_paragraph_lookup,
    build_legislation_references,
    extract_chunk_act_references,
    extract_document_act_references,
    extract_medium_neutral_citations,
    extract_section_references,
    make_legislation_chunks,
    make_judgment_chunks,
    split_legislation_into_sections,
)
import process_corpus


# Paragraph tests protect pinpoint citation safety across NSW text formats.
class ParagraphParsingTests(unittest.TestCase):
    # Neutral citations contain years and court numbers, but those are not
    # paragraph markers.
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

    # Some HTML-to-text conversions restart ordered lists at 1. We preserve
    # source order without treating nested lists as judgment paragraphs.
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

    # Older NSW exports use several marker shapes, including missing spaces.
    def test_parses_glued_markers_without_a_separator(self):
        text = "JUDGMENT\n" + "\n".join(
            f"{number}The court considered the evidence carefully."
            for number in range(1, 26)
        )
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["marker_style"], "glued")
        self.assertEqual(len(parsed["paragraphs"]), 25)
        self.assertEqual(parsed["paragraphs"][0]["text"], "[1] The court considered the evidence carefully.")

    def test_parses_legacy_unindented_plain_markers(self):
        text = "JUDGMENT\n" + "\n".join(
            f"{number} HIS HONOUR: Paragraph text for legacy exports."
            for number in range(1, 12)
        )
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["marker_style"], "plain_unindented")
        self.assertEqual(parsed["status"], "original")
        self.assertEqual(len(parsed["paragraphs"]), 11)

    def test_parses_dot_glued_markers(self):
        text = "JUDGMENT\n" + "\n".join(
            f"{number}.This paragraph lost the space after the marker."
            for number in range(1, 12)
        )
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["marker_style"], "dot_glued")
        self.assertEqual(parsed["status"], "original")
        self.assertEqual(len(parsed["paragraphs"]), 11)

    # Header/order numbers can look like paragraphs; only the body sequence
    # should become citeable.
    def test_ignores_numbered_header_material_before_judgment_heading(self):
        text = """Catchwords: WORKERS COMPENSATION
 2 Section 151Z issue noted in catchwords.
JUDGMENT
 1 HIS HONOUR: First real paragraph.
 2 Second real paragraph.
 3 Third real paragraph."""
        parsed = parse_paragraphs(text)
        self.assertEqual(
            [item["paragraph_number"] for item in parsed["paragraphs"]],
            [1, 2, 3],
        )
        self.assertTrue(parsed["paragraphs"][0]["text"].startswith("[1] HIS HONOUR"))

    def test_later_judgment_heading_does_not_trim_existing_body(self):
        text = """Judgment
1. First source paragraph.
2. Second source paragraph.
3. Third source paragraph.
Reasons
1. Restarted list caused by extracted ordered-list markup.
2. More text.
Judgment
1. Another internal restarted list.
2. More internal text.
3. More internal text."""
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "reconstructed")
        self.assertEqual(len(parsed["paragraphs"]), 8)
        self.assertEqual(parsed["paragraphs"][0]["paragraph_number"], 1)
        self.assertEqual(parsed["paragraphs"][-1]["paragraph_number"], 8)

    # Tchadovitch/Jeffs-style exports change indentation mid-judgment.
    def test_recovers_legacy_sequence_across_indent_changes(self):
        first_run = "\n".join(
            f"  {number} Paragraph text at first indent."
            for number in range(1, 6)
        )
        second_run = "\n".join(
            f"      {number} Paragraph text at second indent."
            for number in range(6, 16)
        )
        text = f"JUDGMENT\n{first_run}\n{second_run}"
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "original")
        self.assertEqual(len(parsed["paragraphs"]), 15)
        self.assertEqual(parsed["paragraphs"][0]["paragraph_number"], 1)
        self.assertEqual(parsed["paragraphs"][-1]["paragraph_number"], 15)

    def test_cross_indent_recovery_skips_intervening_numbered_quotes(self):
        text = """JUDGMENT
  1 First paragraph.
  2 Second paragraph.
                  43 Quoted table row, not a source paragraph.
  3 Third paragraph.
      4 Fourth paragraph after indentation changes.
      5 Fifth paragraph.
      6 Sixth paragraph.
      7 Seventh paragraph.
      8 Eighth paragraph.
      9 Ninth paragraph.
      10 Tenth paragraph."""
        parsed = parse_paragraphs(text)
        self.assertEqual(
            [item["paragraph_number"] for item in parsed["paragraphs"]],
            list(range(1, 11)),
        )

    def test_ignores_trailing_orders_list_in_unnumbered_judgment(self):
        # Do not cite a trailing orders list as if it were the judgment body.
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
        # A tiny marker group must not swallow a long unnumbered body.
        text = """Medium Neutral Citation: Example v Person [2020] NSWDC 4
Decision:                 2. Judgment for the defendant.
                          3. The exhibits may be returned.
                          4. Liberty to apply on 7 days notice.
""" + "The court reviewed the medical evidence in detail. " * 800
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "unavailable")
        self.assertEqual(parsed["paragraphs"], [])

    def test_rejects_pdf_only_table_of_contents_as_paragraphs(self):
        text = """NOTE: This decision contains over 1900 paragraphs and images.
Because of its size the decision has been published via a PDF document.
1. Introduction [1]
2. Background [12]
3. Liability [40]
4. Causation [67]
5. Damages [90]
6. Orders [120]
7. Acronyms [130]
8. Names [140]
9. Amendments [150]
10. Disclaimer [160]"""
        parsed = parse_paragraphs(text)
        self.assertEqual(parsed["status"], "unavailable")
        self.assertEqual(parsed["paragraphs"], [])

    # A valid excerpt can begin above paragraph 1, so long monotonic runs are
    # allowed when they look like the real body.
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


# Header tests keep the structured citation blocks available for linking.
class HeaderParsingTests(unittest.TestCase):
    # Header lists are the cleanest source for cited cases and legislation.
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


class CaseReferenceTests(unittest.TestCase):
    def setUp(self):
        self.baseline_docs = [
            {
                "version_id": "case_nswca_20",
                "citation": "Example v Person [2018] NSWCA 20",
                "parsed_court": "NSWCA",
                "date": "2018-05-01",
                "url": "https://example.test/case_nswca_20",
            }
        ]
        self.baseline = build_baseline_case_index(self.baseline_docs)
        self.paragraphs = [
            {
                "paragraph_number": 12,
                "text": "[12] Strong v Woolworths Ltd was discussed.",
            },
            {
                "paragraph_number": 13,
                "text": "[13] Example v Person [2018] NSWCA 20 was followed.",
            },
        ]
        self.paragraph_lookup = build_paragraph_lookup(self.paragraphs)

    def test_extracts_medium_neutral_citation_parts(self):
        citations = extract_medium_neutral_citations(
            "Strong v Woolworths Ltd [2012] HCA 5"
        )
        self.assertEqual(citations[0]["citation"], "[2012] HCA 5")
        self.assertEqual(citations[0]["court"], "HCA")
        self.assertEqual(citations[0]["year"], 2012)

    def test_marks_high_court_case_outside_baseline(self):
        refs = build_case_references(
            ["Strong v Woolworths Ltd [2012] HCA 5"],
            self.baseline,
            self.paragraph_lookup,
        )
        self.assertEqual(refs[0]["baseline_status"], "outside_baseline")
        self.assertFalse(refs[0]["in_baseline"])
        self.assertFalse(refs[0]["full_text_available"])
        self.assertEqual(refs[0]["discussing_paragraph_numbers"], [12])

    def test_matches_actual_indexed_judgment(self):
        refs = build_case_references(
            ["Example v Person [2018] NSWCA 20"],
            self.baseline,
            self.paragraph_lookup,
        )
        self.assertEqual(refs[0]["baseline_status"], "in_baseline")
        self.assertTrue(refs[0]["baseline_candidate"])
        self.assertEqual(refs[0]["matching_version_id"], "case_nswca_20")
        self.assertEqual(refs[0]["discussing_paragraph_numbers"], [13])

    def test_uses_matched_neutral_citation_as_primary(self):
        refs = build_case_references(
            ["Parallel citation [2004] HCA 1; Example v Person [2018] NSWCA 20"],
            self.baseline,
            self.paragraph_lookup,
        )
        self.assertEqual(refs[0]["baseline_status"], "in_baseline")
        self.assertTrue(refs[0]["baseline_candidate"])
        self.assertEqual(refs[0]["parsed_court"], "NSWCA")
        self.assertEqual(refs[0]["parsed_year"], 2018)

    def test_nsw_case_can_be_eligible_but_not_indexed(self):
        refs = build_case_references(
            ["Missing v Case [2018] NSWSC 999"],
            self.baseline,
            self.paragraph_lookup,
        )
        self.assertEqual(refs[0]["baseline_status"], "eligible_but_not_indexed")
        self.assertTrue(refs[0]["baseline_candidate"])
        self.assertFalse(refs[0]["in_baseline"])
        self.assertIsNone(refs[0]["matching_version_id"])

    def test_unclassified_without_medium_neutral_citation(self):
        refs = build_case_references(
            ["Brodie v Singleton Shire Council (2001) 206 CLR 512"],
            self.baseline,
            self.paragraph_lookup,
        )
        self.assertEqual(refs[0]["baseline_status"], "unclassified")
        self.assertIsNone(refs[0]["parsed_court"])
        self.assertIsNone(refs[0]["parsed_year"])


# Legislation tests protect Act section parsing and bounded retrieval chunks.
class LegislationParsingTests(unittest.TestCase):
    # Schedule clauses can reuse section numbers, so IDs must include context.
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

    def test_long_legislation_provisions_are_fragmented(self):
        provision = {
            "provision_id": "s_5D",
            "provision_type": "section",
            "provision": "s 5D",
            "section": "5D",
            "schedule": None,
            "clause": None,
            "heading": "General principles",
            "part": "1A",
            "part_heading": "Part 1A Negligence",
            "division": "3",
            "division_heading": "Division 3 Causation",
            "schedule_heading": None,
            "text": "5D General principles\n" + ("Sentence text. " * 700),
        }
        chunks = make_legislation_chunks(provision)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["text"]) <= MAX_CHUNK_CHARS for chunk in chunks))
        self.assertEqual(chunks[0]["provision_fragment"]["index"], 1)
        self.assertEqual(chunks[-1]["provision_fragment"]["count"], len(chunks))


# Act reference tests keep NSW Civil Liability Act sections tied to the right Act.
class ActReferenceTests(unittest.TestCase):
    def setUp(self):
        self.order = [
            "1", "3", "3B", "5B", "5C", "5D", "5E",
            "13", "15B", "43", "56", "58", "64", "69",
        ]
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

    # Shorthand is safe only when the document header already ties it to the
    # target Act.
    def test_rejects_foreign_act_shorthand(self):
        sections, _ = extract_chunk_act_references(
            "Section 56 of the Civil Procedure Act applies. "
            "The court also considered s 5D.",
            ["5D"],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, ["5D"])

    def test_other_act_reference_overrides_document_level_sections(self):
        sections, _ = extract_chunk_act_references(
            "Section 64 of the Civil Procedure Act 2005 (NSW) provides the "
            "power to amend the pleadings.",
            ["64"],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, [])

    def test_mixed_chunk_keeps_legitimate_cla_sections(self):
        sections, _ = extract_chunk_act_references(
            "The dispute was referred to in s 58 of the Motor Accidents "
            "Compensation Act 1999 (NSW). Sections 3B and 15B of the "
            "Civil Liability Act 2002 (NSW) are separate issues.",
            ["3B", "15B", "58"],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, ["3B", "15B"])

    def test_separate_other_act_section_does_not_mask_cla_section(self):
        sections, _ = extract_chunk_act_references(
            "Some persons are entitled to a pension under s 43 of the Social "
            "Security Act 1991. Section 13 of the Civil Liability Act 2002 "
            "(NSW) remains relevant.",
            ["13", "43"],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, ["13"])

    def test_that_act_reference_keeps_other_act_context(self):
        sections, _ = extract_chunk_act_references(
            "Section 69 of the Evidence Act relevantly provides the rule. "
            "The opinion was excluded by s 69(3) of that Act.",
            ["69"],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, [])

    def test_later_same_section_uses_prior_other_act_context(self):
        sections, _ = extract_chunk_act_references(
            "The evidence was admissible under s 69(2)(a) of the Evidence Act. "
            "That means s 69 still applied to the document.",
            ["69"],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, [])

    def test_accepts_explicit_civil_liability_act_reference(self):
        sections, _ = extract_chunk_act_references(
            "Section 56 of the Civil Liability Act 2002 applies.",
            [],
            self.valid,
            self.order,
        )
        self.assertEqual(sections, ["56"])

    def test_keeps_sections_attached_to_their_legislation_entry(self):
        cited = [
            "Civil Liability Act 1936 (SA), s 34",
            "Civil Liability Act 2002 (NSW), s 5D",
        ]
        references = build_legislation_references(cited, self.valid | {"34"}, self.order)
        self.assertEqual(references[0]["section_references"], ["34"])
        self.assertFalse(references[0]["is_civil_liability_act_2002_nsw"])
        self.assertEqual(references[0]["civil_liability_act_sections"], [])
        self.assertEqual(references[1]["section_references"], ["5D"])
        self.assertTrue(references[1]["is_civil_liability_act_2002_nsw"])
        self.assertEqual(references[1]["civil_liability_act_sections"], ["5D"])

        sections, _, structured = extract_document_act_references(
            cited, self.valid | {"34"}, self.order
        )
        self.assertEqual(sections, ["5D"])
        self.assertEqual(structured, references)

    # Real corpus headers contain a few odd-but-valid spellings of the 2002 Act.
    def test_accepts_parenthesised_year_for_target_act(self):
        cited = ["Civil Liability Act (2002), s5B, s5D"]
        sections, _, structured = extract_document_act_references(
            cited, self.valid, self.order
        )
        self.assertEqual(sections, ["5B", "5D"])
        self.assertTrue(structured[0]["is_civil_liability_act_2002_nsw"])

    def test_accepts_comma_and_jurisdiction_before_year_for_target_act(self):
        cited = [
            "Civil Liability Act, 2002, s 5B",
            "Civil Liability Act (NSW) 2002, s 5D",
        ]
        sections, _, structured = extract_document_act_references(
            cited, self.valid, self.order
        )
        self.assertEqual(sections, ["5B", "5D"])
        self.assertTrue(all(item["is_civil_liability_act_2002_nsw"] for item in structured))

    def test_rejects_typo_year_civil_liability_acts(self):
        cited = [
            "Civil Liability Act 2000 (NSW), s 5D",
            "Civil Liability Act 2022 (NSW), s 16",
        ]
        sections, _, structured = extract_document_act_references(
            cited, self.valid, self.order
        )
        self.assertEqual(sections, [])
        self.assertTrue(
            all(not item["is_civil_liability_act_2002_nsw"] for item in structured)
        )


# Output validation tests exercise the generated JSONL invariants.
class ProcessedOutputValidationTests(unittest.TestCase):
    # The validator checks the invariants the UI relies on when creating links.
    def test_validate_processed_outputs_accepts_well_formed_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            old_paths = (
                process_corpus.OUTPUT_JUDGMENTS,
                process_corpus.OUTPUT_JUDGMENT_METADATA,
                process_corpus.OUTPUT_LEGISLATION,
            )
            try:
                process_corpus.OUTPUT_JUDGMENTS = temp_path / "judgment_chunks.jsonl"
                process_corpus.OUTPUT_JUDGMENT_METADATA = (
                    temp_path / "judgment_metadata.jsonl"
                )
                process_corpus.OUTPUT_LEGISLATION = temp_path / "legislation_chunks.jsonl"

                process_corpus.OUTPUT_LEGISLATION.write_text(
                    '{"chunk_id":"act_s_5D","document_type":"legislation",'
                    '"provision_type":"section","section":"5D","text":"5D Text"}\n',
                    encoding="utf-8",
                )
                process_corpus.OUTPUT_JUDGMENT_METADATA.write_text(
                    '{"version_id":"case_1",'
                    '"cases_cited":[],'
                    '"case_references":[],'
                    '"civil_liability_act_sections":["5D"],'
                    '"legislation_references":[{'
                    '"text":"Civil Liability Act 2002 (NSW), s 5D",'
                    '"section_references":["5D"],'
                    '"part_references":[],'
                    '"is_civil_liability_act_2002_nsw":true,'
                    '"civil_liability_act_sections":["5D"],'
                    '"civil_liability_act_parts":[]'
                    '}]}\n',
                    encoding="utf-8",
                )
                process_corpus.OUTPUT_JUDGMENTS.write_text(
                    '{"chunk_id":"case_1_chunk_1","document_type":"judgment",'
                    '"version_id":"case_1","citation_available":true,'
                    '"paragraph_start":1,"paragraph_end":2,'
                    '"paragraph_numbers":[1,2],"legislation_sections":["5D"],'
                    '"document_legislation_sections":["5D"],"text":"[1] A\\n[2] B"}\n',
                    encoding="utf-8",
                )

                result = process_corpus.validate_processed_outputs(["5D"])
            finally:
                (
                    process_corpus.OUTPUT_JUDGMENTS,
                    process_corpus.OUTPUT_JUDGMENT_METADATA,
                    process_corpus.OUTPUT_LEGISLATION,
                ) = old_paths

        self.assertTrue(result["valid"])
        self.assertEqual(result["judgment_chunks"], 1)
        self.assertEqual(result["judgment_metadata"], 1)
        self.assertEqual(result["legislation_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
