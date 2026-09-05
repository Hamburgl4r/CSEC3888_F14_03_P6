"""Build retrieval-ready judgment and legislation records.

Run from the repository root:

    python3 scripts/process_corpus.py

The pipeline preserves source traceability, bounds every retrieval chunk, and
never labels an unnumbered passage with an invented paragraph number.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from parse_judgements import parse_judgment


JUDGMENTS_FILE = Path("data/filtered/judgments.jsonl")
ACT_FILE = Path("data/filtered/civil_liability_act.json")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JUDGMENTS = OUTPUT_DIR / "judgment_chunks.jsonl"
OUTPUT_JUDGMENT_METADATA = OUTPUT_DIR / "judgment_metadata.jsonl"
OUTPUT_LEGISLATION = OUTPUT_DIR / "legislation_chunks.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "processing_report.json"

MAX_CHUNK_CHARS = 3000

# Citation patterns used throughout the processor. Keep these near the top so
# changes to recognised courts, Acts, or section syntax are easy to audit.
TARGET_ACT_RE = re.compile(
    r"\bCivil\s+Liability\s+Act\b\s*(?:,\s*)?"
    r"(?:(?:\(\s*NSW\s*\)\s*)?(?:2002|\(\s*2002\s*\))|"
    r"(?:2002|\(\s*2002\s*\))\s*(?:\(\s*NSW\s*\))?)",
    re.I,
)
ACT_JURISDICTION_RE = re.compile(r"^\s*\(([^)]+)\)")
SECTION_REFERENCE_RE = re.compile(
    # ``s 5D``, ``ss 5B-5D`` and the unspaced ``s5D`` all occur.
    r"\b(?:ss?|sections?)\s*"
    r"(\d+[A-Za-z]*(?:\([^)\s]+\))*"
    r"(?:\s*(?:,|and|to|[-\u2013\u2014])\s*"
    r"\d+[A-Za-z]*(?:\([^)\s]+\))*)*)",
    re.I,
)
PART_REFERENCE_RE = re.compile(r"\b(?:Pt|Part)\s+(\d+[A-Za-z]*)\b", re.I)
# A subsection number in ``5D(1)(a)`` is not another section reference.
SECTION_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9(])\d+[A-Za-z]*")
RAW_SECTION_REFERENCE_RE = re.compile(
    r"\b(?:ss?|sections?)\s*"
    r"(\d+[A-Za-z]*(?:\([^)\s]+\))*"
    r"(?:\s*(?:,|and|to|[-\u2013\u2014])\s*"
    r"\d+[A-Za-z]*(?:\([^)\s]+\))*)*)",
    re.I,
)
ACT_AFTER_SECTION_RE = re.compile(
    r"^\s*(?:\([^)\s]+\))*\s+of\s+(?:the\s+)?"
    r"(?P<title>[A-Z][A-Za-z0-9&'’(),/-]*"
    r"(?:\s+[A-Za-z0-9&'’(),/-]+){0,12}\s+Act"
    r"(?:\s+\d{4})?(?:\s*\([^)]+\))?)"
)
ACT_PRONOUN_AFTER_SECTION_RE = re.compile(
    r"^\s*(?:\([^)\s]+\))*\s+of\s+(?:that|the)\s+Act\b", re.I
)
MEDIUM_NEUTRAL_CITATION_RE = re.compile(
    r"\[(?P<year>\d{4})\]\s+(?P<court>[A-Z][A-Z0-9]{1,12})\s+(?P<number>\d+)"
)
BASELINE_CASE_COURTS = {"NSWSC", "NSWCA", "NSWDC"}
CASE_NAME_STOP_RE = re.compile(r"\s+(?:\[\d{4}\]|\(\d{4}\)|\(\d{1,3}\s)", re.I)


# ---------------------------------------------------------------------------
# Shared text and citation helpers


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def normalize_lookup_text(text: str) -> str:
    """Normalise strings for citation matching, not for display."""
    return re.sub(r"\s+", " ", text or "").strip().upper()


def extract_medium_neutral_citations(text: str) -> list[dict[str, Any]]:
    """Extract medium-neutral citation tokens such as ``[2012] HCA 5``."""
    citations = []
    seen = set()
    for match in MEDIUM_NEUTRAL_CITATION_RE.finditer(text or ""):
        citation = normalize_lookup_text(match.group(0))
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(
            {
                "citation": citation,
                "year": int(match.group("year")),
                "court": match.group("court").upper(),
                "number": int(match.group("number")),
            }
        )
    return citations


def _case_name_search_key(citation: str) -> str:
    """Use the leading case name as a fallback when no neutral cite appears."""
    name = CASE_NAME_STOP_RE.split(citation or "", maxsplit=1)[0]
    return normalize_lookup_text(name)


def build_baseline_case_index(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index the judgments that are actually present in the retrieval corpus."""
    index: dict[str, dict[str, Any]] = {}
    for doc in docs:
        for neutral in extract_medium_neutral_citations(doc.get("citation") or ""):
            index.setdefault(
                neutral["citation"],
                {
                    "version_id": doc.get("version_id"),
                    "citation": doc.get("citation"),
                    "court": doc.get("parsed_court"),
                    "date": doc.get("date"),
                    "url": doc.get("url"),
                },
            )
    return index


def find_case_discussion_paragraphs(
    cited_case: str,
    medium_neutral_citations: list[dict[str, Any]],
    paragraph_lookup: list[tuple[int, str]],
) -> list[int]:
    """Find source paragraphs in this judgment that mention a cited case."""
    needles = [item["citation"] for item in medium_neutral_citations]
    case_name = _case_name_search_key(cited_case)
    if len(case_name) >= 12:
        needles.append(case_name)

    paragraph_numbers = []
    for paragraph_number, paragraph_text in paragraph_lookup:
        if any(needle and needle in paragraph_text for needle in needles):
            paragraph_numbers.append(paragraph_number)
    return paragraph_numbers


def build_paragraph_lookup(paragraphs: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Pre-normalise paragraph text once so case-reference lookup stays fast."""
    return [
        (
            paragraph["paragraph_number"],
            normalize_lookup_text(paragraph.get("text") or ""),
        )
        for paragraph in paragraphs
    ]


def build_case_references(
    cases_cited: list[str],
    baseline_case_index: dict[str, dict[str, Any]],
    paragraph_lookup: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Classify cited cases against the actual indexed judgment baseline.

    A NSWSC/NSWCA/NSWDC citation from 2010 onward is only a candidate. It is
    marked ``in_baseline`` only if its neutral citation exists in this corpus.
    """
    references = []
    for cited_case in cases_cited:
        neutral_citations = extract_medium_neutral_citations(cited_case)
        first_neutral = neutral_citations[0] if neutral_citations else None
        baseline_match = None
        for neutral in neutral_citations:
            baseline_match = baseline_case_index.get(neutral["citation"])
            if baseline_match:
                break

        baseline_candidate = bool(
            first_neutral
            and first_neutral["court"] in BASELINE_CASE_COURTS
            and first_neutral["year"] >= 2010
        )
        # This split prevents evaluation from treating HCA/FCA/older NSW cases
        # as retrieval failures when their full text is intentionally absent.
        if baseline_match:
            baseline_status = "in_baseline"
        elif first_neutral and baseline_candidate:
            baseline_status = "eligible_but_not_indexed"
        elif first_neutral:
            baseline_status = "outside_baseline"
        else:
            baseline_status = "unclassified"

        references.append(
            {
                "text": cited_case,
                "medium_neutral_citations": neutral_citations,
                "parsed_court": first_neutral["court"] if first_neutral else None,
                "parsed_year": first_neutral["year"] if first_neutral else None,
                "baseline_candidate": baseline_candidate,
                "in_baseline": baseline_match is not None,
                "baseline_status": baseline_status,
                "matching_version_id": (
                    baseline_match["version_id"] if baseline_match else None
                ),
                "matching_citation": (
                    baseline_match["citation"] if baseline_match else None
                ),
                "full_text_available": baseline_match is not None,
                "discussing_paragraph_numbers": find_case_discussion_paragraphs(
                    cited_case, neutral_citations, paragraph_lookup
                ),
            }
        )
    return references


# ---------------------------------------------------------------------------
# Chunk construction


def _split_at_boundary(text: str, limit: int) -> tuple[str, str]:
    """Split once, preferring paragraph, sentence, then word boundaries."""
    if len(text) <= limit:
        return text, ""

    minimum = max(1, limit // 2)
    split_at = text.rfind("\n\n", minimum, limit + 1)
    if split_at < minimum:
        sentence_matches = list(re.finditer(r"(?<=[.!?])\s+", text[minimum:limit]))
        split_at = (
            minimum + sentence_matches[-1].end()
            if sentence_matches
            else text.rfind(" ", minimum, limit + 1)
        )
    if split_at < minimum:
        split_at = limit

    return text[:split_at].rstrip(), text[split_at:].lstrip()


def split_text_bounded(
    text: str, limit: int = MAX_CHUNK_CHARS, prefix: str = ""
) -> list[str]:
    """Split text into non-empty pieces no longer than ``limit`` characters."""
    text = clean_text(text)
    if not text:
        return []
    if len(prefix) >= limit:
        raise ValueError("Chunk prefix must be shorter than the chunk limit")

    pieces = []
    remaining = text
    content_limit = limit - len(prefix)
    while remaining:
        piece, remaining = _split_at_boundary(remaining, content_limit)
        pieces.append(prefix + piece)
    return pieces


def _fragment_paragraph(paragraph: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one source paragraph while preserving its pinpoint identity."""
    number = paragraph["paragraph_number"]
    text = paragraph["text"]
    canonical_prefix = f"[{number}] "
    if text.startswith(canonical_prefix):
        text = text[len(canonical_prefix):]

    pieces = split_text_bounded(text, MAX_CHUNK_CHARS, canonical_prefix)
    return [
        {
            "text": piece,
            "paragraph_number": number,
            "fragment_index": index,
            "fragment_count": len(pieces),
        }
        for index, piece in enumerate(pieces, start=1)
    ]


def make_judgment_chunks(
    doc: dict[str, Any], parsed: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Create bounded paragraph-aware chunks for one judgment."""
    text = doc.get("text") or ""
    parsed = parsed or parse_judgment(text)
    paragraphs = parsed["paragraphs"]

    if not paragraphs:
        # These passages are searchable, but the application must not present
        # them as pinpoint citations because the source paragraph is unknown.
        return [
            {
                "text": piece,
                "paragraph_start": None,
                "paragraph_end": None,
                "paragraph_numbers": [],
                "paragraph_numbering": "unavailable",
                "citation_available": False,
                "paragraph_fragment": None,
            }
            for piece in split_text_bounded(text)
        ]

    units = []
    for paragraph in paragraphs:
        units.extend(_fragment_paragraph(paragraph))

    chunks: list[dict[str, Any]] = []
    current_units: list[dict[str, Any]] = []
    current_length = 0

    def save_current() -> None:
        nonlocal current_units, current_length
        if not current_units:
            return
        # ``paragraph_numbers`` preserves every source paragraph covered by the
        # chunk, while start/end support simpler range-style displays.
        numbers = []
        for unit in current_units:
            if unit["paragraph_number"] not in numbers:
                numbers.append(unit["paragraph_number"])
        only_unit = current_units[0] if len(current_units) == 1 else None
        fragment = None
        if only_unit and only_unit["fragment_count"] > 1:
            fragment = {
                "index": only_unit["fragment_index"],
                "count": only_unit["fragment_count"],
            }
        chunks.append(
            {
                "text": "\n".join(unit["text"] for unit in current_units),
                "paragraph_start": numbers[0],
                "paragraph_end": numbers[-1],
                "paragraph_numbers": numbers,
                "paragraph_numbering": parsed["status"],
                "citation_available": True,
                "paragraph_fragment": fragment,
            }
        )
        current_units = []
        current_length = 0

    for unit in units:
        separator_length = 1 if current_units else 0
        if (
            current_units
            and current_length + separator_length + len(unit["text"])
            > MAX_CHUNK_CHARS
        ):
            save_current()

        # Fragments of one oversized paragraph remain separate so each chunk
        # can expose its fragment index.
        if unit["fragment_count"] > 1:
            save_current()
            current_units = [unit]
            current_length = len(unit["text"])
            save_current()
        else:
            current_units.append(unit)
            current_length += separator_length + len(unit["text"])

    save_current()
    return chunks


# ---------------------------------------------------------------------------
# Civil Liability Act reference extraction


def _expand_section_range(
    start: str, end: str, section_order: list[str]
) -> list[str]:
    start = start.upper()
    end = end.upper()
    lookup = {section.upper(): index for index, section in enumerate(section_order)}
    if start not in lookup or end not in lookup or lookup[start] > lookup[end]:
        return [start, end]
    # Avoid turning a malformed reference into a very large accidental range.
    if lookup[end] - lookup[start] > 25:
        return [start, end]
    return section_order[lookup[start]:lookup[end] + 1]


def extract_section_references(
    text: str, valid_sections: set[str], section_order: list[str]
) -> list[str]:
    """Extract base section identifiers from ``s``, ``ss`` and ranges."""
    references: set[str] = set()
    valid_lookup = {section.upper(): section for section in valid_sections}

    for match in SECTION_REFERENCE_RE.finditer(text):
        expression = match.group(1)
        tokens = [token.upper() for token in SECTION_TOKEN_RE.findall(expression)]
        if not tokens:
            continue

        is_range = bool(re.search(r"\bto\b|[-\u2013\u2014]", expression, re.I))
        if is_range and len(tokens) >= 2:
            expanded = _expand_section_range(tokens[0], tokens[-1], section_order)
            tokens = expanded + tokens[1:-1]

        for token in tokens:
            base = re.match(r"\d+[A-Z]*", token).group(0)
            if base in valid_lookup:
                references.add(valid_lookup[base])

    order_lookup = {section: index for index, section in enumerate(section_order)}
    return sorted(references, key=lambda section: order_lookup.get(section, 10**9))


def _section_reference_names_non_target_act(text: str, match: re.Match[str]) -> bool:
    """Return true when this local section citation is tied to another Act."""
    suffix = text[match.end():match.end() + 180]
    act_match = ACT_AFTER_SECTION_RE.match(suffix)
    if not act_match:
        return False
    return not is_target_civil_liability_act(act_match.group("title"))


def _section_reference_names_target_act(text: str, match: re.Match[str]) -> bool:
    suffix = text[match.end():match.end() + 180]
    act_match = ACT_AFTER_SECTION_RE.match(suffix)
    return bool(act_match and is_target_civil_liability_act(act_match.group("title")))


def _section_bases_from_match(match: re.Match[str]) -> set[str]:
    bases = set()
    for token in SECTION_TOKEN_RE.findall(match.group(1)):
        base_match = re.match(r"\d+[A-Za-z]*", token)
        if base_match:
            bases.add(base_match.group(0).upper())
    return bases


def _previous_same_section_named_non_target(
    text: str, match: re.Match[str], lookback: int = MAX_CHUNK_CHARS
) -> bool:
    """Carry forward local Act context for repeated references like ``s 69``."""
    bases = _section_bases_from_match(match)
    if not bases:
        return False

    start = max(0, match.start() - lookback)
    prefix = text[start:match.start()]
    for prior in SECTION_REFERENCE_RE.finditer(prefix):
        if bases & _section_bases_from_match(prior):
            if _section_reference_names_non_target_act(prefix, prior):
                return True
    return False


def _section_reference_is_locally_non_target(
    text: str, match: re.Match[str]
) -> bool:
    """Detect section mentions that should not be treated as CLA references."""
    if _section_reference_names_non_target_act(text, match):
        return True
    if _section_reference_names_target_act(text, match):
        return False

    suffix = text[match.end():match.end() + 80]
    uses_act_pronoun = bool(ACT_PRONOUN_AFTER_SECTION_RE.match(suffix))
    if uses_act_pronoun and _previous_same_section_named_non_target(text, match):
        return True

    if _previous_same_section_named_non_target(text, match):
        nearby = text[max(0, match.start() - 180):match.end() + 180]
        return not TARGET_ACT_RE.search(nearby)
    return False


def extract_civil_liability_section_references(
    text: str,
    valid_sections: set[str],
    section_order: list[str],
) -> list[str]:
    """Extract section references unless the local wording names another Act."""
    references: set[str] = set()
    for match in SECTION_REFERENCE_RE.finditer(text):
        if _section_reference_is_locally_non_target(text, match):
            continue
        references.update(
            extract_section_references(match.group(0), valid_sections, section_order)
        )

    order_lookup = {section: index for index, section in enumerate(section_order)}
    return sorted(references, key=lambda section: order_lookup.get(section, 10**9))


def extract_raw_section_references(text: str) -> list[str]:
    """Extract section identifiers from one legislation citation string."""
    references: list[str] = []
    for match in RAW_SECTION_REFERENCE_RE.finditer(text):
        for token in SECTION_TOKEN_RE.findall(match.group(1)):
            base_match = re.match(r"\d+[A-Za-z]*", token)
            if not base_match:
                continue
            base = base_match.group(0)
            if base not in references:
                references.append(base)
    return references


def is_target_civil_liability_act(text: str) -> bool:
    """Return true for the NSW Civil Liability Act 2002 citation entry."""
    for match in TARGET_ACT_RE.finditer(text):
        suffix = text[match.end():match.end() + 40]
        jurisdiction = ACT_JURISDICTION_RE.match(suffix)
        if jurisdiction and jurisdiction.group(1).strip().upper() != "NSW":
            continue
        return True
    return False


def build_legislation_references(
    legislation_cited: list[str],
    valid_sections: set[str],
    section_order: list[str],
) -> list[dict[str, Any]]:
    """Keep section references attached to the legislation entry they came from."""
    references = []
    for item in legislation_cited:
        is_target_act = is_target_civil_liability_act(item)
        references.append(
            {
                "text": item,
                "section_references": extract_raw_section_references(item),
                "part_references": sorted(set(PART_REFERENCE_RE.findall(item))),
                "is_civil_liability_act_2002_nsw": is_target_act,
                "civil_liability_act_sections": (
                    extract_section_references(item, valid_sections, section_order)
                    if is_target_act
                    else []
                ),
                "civil_liability_act_parts": (
                    sorted(set(PART_REFERENCE_RE.findall(item)))
                    if is_target_act
                    else []
                ),
            }
        )
    return references


def extract_document_act_references(
    legislation_cited: list[str],
    valid_sections: set[str],
    section_order: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Summarise target Act references from the judgment header only."""
    sections: set[str] = set()
    parts: set[str] = set()
    legislation_references = build_legislation_references(
        legislation_cited, valid_sections, section_order
    )
    for item in legislation_references:
        if not item["is_civil_liability_act_2002_nsw"]:
            continue
        sections.update(item["civil_liability_act_sections"])
        parts.update(item["civil_liability_act_parts"])
    return (
        [section for section in section_order if section in sections],
        sorted(parts),
        legislation_references,
    )


def extract_chunk_act_references(
    text: str,
    document_sections: list[str],
    valid_sections: set[str],
    section_order: list[str],
) -> tuple[list[str], list[str]]:
    """Conservatively identify Civil Liability Act references in a passage.

    Shorthand references such as ``s 5D`` are accepted only when that section
    is listed for this Act in the judgment header. References close to an
    explicit mention of the Act are accepted even when the header is missing.
    """
    # Start with document-level shorthand, then add local explicit CLA mentions.
    # Each pass uses the same guard against "s 64 of the Civil Procedure Act".
    sections = set(
        extract_civil_liability_section_references(
            text, set(document_sections), section_order
        )
    )
    parts: set[str] = set()

    for act_match in TARGET_ACT_RE.finditer(text):
        suffix = text[act_match.end():act_match.end() + 40]
        jurisdiction = ACT_JURISDICTION_RE.match(suffix)
        if jurisdiction and jurisdiction.group(1).strip().upper() != "NSW":
            continue
        start = max(
            0,
            text.rfind("\n", 0, act_match.start()),
            text.rfind(".", 0, act_match.start()),
            act_match.start() - 250,
        )
        possible_ends = [
            position
            for position in (
                text.find("\n", act_match.end()),
                text.find(".", act_match.end()),
                act_match.end() + 250,
            )
            if position != -1
        ]
        end = min(possible_ends) if possible_ends else len(text)
        window = text[start:end]
        sections.update(
            extract_civil_liability_section_references(
                window, valid_sections, section_order
            )
        )
        parts.update(PART_REFERENCE_RE.findall(window))

    return (
        [section for section in section_order if section in sections],
        sorted(parts),
    )


# ---------------------------------------------------------------------------
# Legislation parsing and output


def split_legislation_into_sections(text: str) -> list[dict[str, Any]]:
    """Parse main Act sections and schedule clauses while retaining hierarchy."""
    lines = text.splitlines()
    provisions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_lines: list[str] = []
    schedule: str | None = None
    schedule_heading: str | None = None
    part: str | None = None
    part_heading: str | None = None
    division: str | None = None
    division_heading: str | None = None

    def save_current() -> None:
        nonlocal current, current_lines
        if current:
            current["text"] = clean_text("\n".join(current_lines))
            provisions.append(current)
        current = None
        current_lines = []

    for line in lines:
        schedule_match = re.match(
            r"^Schedule\s+(\d+[A-Za-z]*)\s*(.*)$", line
        )
        if schedule_match:
            save_current()
            schedule = schedule_match.group(1)
            schedule_heading = line.strip()
            part = part_heading = division = division_heading = None
            continue

        part_match = re.match(r"^Part\s+(\d+[A-Za-z]*)\s*(.*)$", line)
        if part_match:
            save_current()
            part = part_match.group(1)
            part_heading = line.strip()
            division = division_heading = None
            continue

        division_match = re.match(
            r"^Division\s+(\d+[A-Za-z]*)\s*(.*)$", line
        )
        if division_match:
            save_current()
            division = division_match.group(1)
            division_heading = line.strip()
            continue

        # True provision headings are flush-left in the corpus. Preserving
        # indentation prevents definition text beginning with a year (for
        # example "    2002 amending Act...") from becoming a fake section.
        provision_match = re.match(r"^(\d+[A-Za-z]*)\s+(\S.*)$", line)
        if provision_match:
            save_current()
            number = provision_match.group(1)
            heading = provision_match.group(2).strip()
            is_schedule_clause = schedule is not None
            provision_id = (
                f"sch_{schedule}_cl_{number}"
                if is_schedule_clause
                else f"s_{number}"
            )
            current = {
                "provision_id": provision_id,
                "provision_type": (
                    "schedule_clause" if is_schedule_clause else "section"
                ),
                "provision": (
                    f"Schedule {schedule} cl {number}"
                    if is_schedule_clause
                    else f"s {number}"
                ),
                "section": None if is_schedule_clause else number,
                "schedule": schedule,
                "clause": number if is_schedule_clause else None,
                "heading": heading,
                "part": part,
                "part_heading": part_heading,
                "division": division,
                "division_heading": division_heading,
                "schedule_heading": schedule_heading,
            }
            current_lines = [line]
            continue

        if current is not None:
            current_lines.append(line)

    save_current()
    return provisions


def make_legislation_chunks(provision: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one provision into bounded retrieval chunks."""
    pieces = split_text_bounded(provision["text"])
    if not pieces:
        return []
    return [
        {
            **provision,
            "text": piece,
            "provision_fragment": (
                {"index": index, "count": len(pieces)}
                if len(pieces) > 1
                else None
            ),
        }
        for index, piece in enumerate(pieces, start=1)
    ]


def process_legislation() -> tuple[int, int, list[str]]:
    """Write legislation chunks and return the ordered main-section list."""
    if not ACT_FILE.exists():
        raise FileNotFoundError(f"Civil Liability Act file not found: {ACT_FILE}")

    with ACT_FILE.open("r", encoding="utf-8") as infile:
        doc = json.load(infile)

    provisions = split_legislation_into_sections(doc.get("text") or "")
    main_sections = [
        provision["section"]
        for provision in provisions
        if provision["provision_type"] == "section"
    ]

    total_chunks = 0
    with OUTPUT_LEGISLATION.open("w", encoding="utf-8") as outfile:
        for provision in provisions:
            chunks = make_legislation_chunks(provision)
            for index, chunk in enumerate(chunks, start=1):
                chunk_id = (
                    f"{doc.get('version_id', 'civil_liability_act')}_"
                    f"{provision['provision_id']}"
                )
                if len(chunks) > 1:
                    chunk_id = f"{chunk_id}_fragment_{index}"
                output = {
                    "chunk_id": chunk_id,
                    "document_type": "legislation",
                    "version_id": doc.get("version_id"),
                    "citation": doc.get("citation"),
                    "date": doc.get("date"),
                    "url": doc.get("url"),
                    **chunk,
                }
                if len(output["text"]) > MAX_CHUNK_CHARS:
                    raise AssertionError(
                        f"Legislation chunk exceeds limit: {output['chunk_id']}"
                    )
                outfile.write(json.dumps(output, ensure_ascii=False) + "\n")
                total_chunks += 1

    return len(provisions), total_chunks, main_sections


# ---------------------------------------------------------------------------
# Output validation and judgment processing


def validate_processed_outputs(main_sections: list[str]) -> dict[str, Any]:
    """Check generated JSONL files for citation and chunking invariants."""
    required_files = [
        OUTPUT_JUDGMENTS,
        OUTPUT_JUDGMENT_METADATA,
        OUTPUT_LEGISLATION,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        return {
            "valid": False,
            "errors": [f"Missing output file: {path}" for path in missing_files],
            "error_count": len(missing_files),
            "warnings": [],
            "warning_count": 0,
        }

    errors: list[str] = []
    warnings: list[str] = []
    seen_chunk_ids: set[str] = set()
    metadata_version_ids: set[str] = set()
    judgment_version_ids: set[str] = set()
    matched_case_version_ids: set[str] = set()
    valid_sections = set(main_sections)
    valid_case_statuses = {
        "in_baseline",
        "eligible_but_not_indexed",
        "outside_baseline",
        "unclassified",
    }
    counts: Counter[str] = Counter()

    def add_error(message: str) -> None:
        errors.append(message)

    def read_jsonl(path: Path):
        with path.open("r", encoding="utf-8") as infile:
            for line_number, line in enumerate(infile, start=1):
                if not line.strip():
                    continue
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as error:
                    add_error(f"{path} line {line_number}: invalid JSON ({error})")

    for line_number, record in read_jsonl(OUTPUT_LEGISLATION):
        counts["legislation_chunks"] += 1
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            add_error(f"{OUTPUT_LEGISLATION} line {line_number}: missing chunk_id")
        elif chunk_id in seen_chunk_ids:
            add_error(
                f"{OUTPUT_LEGISLATION} line {line_number}: duplicate chunk_id {chunk_id}"
            )
        else:
            seen_chunk_ids.add(chunk_id)
        if record.get("document_type") != "legislation":
            add_error(f"{OUTPUT_LEGISLATION} line {line_number}: wrong document_type")
        if len(record.get("text") or "") > MAX_CHUNK_CHARS:
            add_error(
                f"{OUTPUT_LEGISLATION} line {line_number}: text exceeds {MAX_CHUNK_CHARS} chars"
            )
        if record.get("provision_type") == "section":
            section = record.get("section")
            if section not in valid_sections:
                add_error(
                    f"{OUTPUT_LEGISLATION} line {line_number}: unknown section {section}"
                )

    for line_number, record in read_jsonl(OUTPUT_JUDGMENT_METADATA):
        counts["judgment_metadata"] += 1
        version_id = record.get("version_id")
        if not version_id:
            add_error(f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: missing version_id")
        elif version_id in metadata_version_ids:
            add_error(
                f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: duplicate metadata version_id {version_id}"
            )
        else:
            metadata_version_ids.add(version_id)
        sections = set(record.get("civil_liability_act_sections") or [])
        unknown_sections = sorted(sections - valid_sections)
        if unknown_sections:
            add_error(
                f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: unknown Act sections {unknown_sections}"
            )
        legislation_references = record.get("legislation_references")
        if not isinstance(legislation_references, list):
            add_error(
                f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: missing legislation_references list"
            )
        else:
            target_sections = set()
            for index, reference in enumerate(legislation_references, start=1):
                if "text" not in reference:
                    add_error(
                        f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: legislation reference {index} missing text"
                    )
                if reference.get("is_civil_liability_act_2002_nsw"):
                    target_sections.update(
                        reference.get("civil_liability_act_sections") or []
                    )
            if target_sections != sections:
                add_error(
                    f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: Act sections do not match structured references"
                )
        cases_cited = record.get("cases_cited") or []
        case_references = record.get("case_references")
        if not isinstance(case_references, list):
            add_error(
                f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: missing case_references list"
            )
        else:
            # Keep raw and structured case citations aligned so downstream
            # evaluation can decide what was retrievable from this corpus.
            if len(case_references) != len(cases_cited):
                add_error(
                    f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: case_references length does not match cases_cited"
                )
            for index, reference in enumerate(case_references, start=1):
                if "text" not in reference:
                    add_error(
                        f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: case reference {index} missing text"
                    )
                if reference.get("baseline_status") not in valid_case_statuses:
                    add_error(
                        f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: case reference {index} has invalid baseline_status"
                    )
                if reference.get("in_baseline"):
                    matching_version_id = reference.get("matching_version_id")
                    if not matching_version_id:
                        add_error(
                            f"{OUTPUT_JUDGMENT_METADATA} line {line_number}: in-baseline case reference {index} missing matching_version_id"
                        )
                    else:
                        matched_case_version_ids.add(matching_version_id)

    for line_number, record in read_jsonl(OUTPUT_JUDGMENTS):
        counts["judgment_chunks"] += 1
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            add_error(f"{OUTPUT_JUDGMENTS} line {line_number}: missing chunk_id")
        elif chunk_id in seen_chunk_ids:
            add_error(
                f"{OUTPUT_JUDGMENTS} line {line_number}: duplicate chunk_id {chunk_id}"
            )
        else:
            seen_chunk_ids.add(chunk_id)

        version_id = record.get("version_id")
        if version_id:
            judgment_version_ids.add(version_id)
        else:
            add_error(f"{OUTPUT_JUDGMENTS} line {line_number}: missing version_id")

        if record.get("document_type") != "judgment":
            add_error(f"{OUTPUT_JUDGMENTS} line {line_number}: wrong document_type")
        if len(record.get("text") or "") > MAX_CHUNK_CHARS:
            add_error(
                f"{OUTPUT_JUDGMENTS} line {line_number}: text exceeds {MAX_CHUNK_CHARS} chars"
            )

        paragraph_numbers = record.get("paragraph_numbers") or []
        if record.get("citation_available"):
            if not paragraph_numbers:
                add_error(
                    f"{OUTPUT_JUDGMENTS} line {line_number}: citation_available without paragraph_numbers"
                )
            elif record.get("paragraph_start") != paragraph_numbers[0]:
                add_error(
                    f"{OUTPUT_JUDGMENTS} line {line_number}: paragraph_start does not match first paragraph"
                )
            elif record.get("paragraph_end") != paragraph_numbers[-1]:
                add_error(
                    f"{OUTPUT_JUDGMENTS} line {line_number}: paragraph_end does not match last paragraph"
                )
        elif (
            record.get("paragraph_start") is not None
            or record.get("paragraph_end") is not None
            or paragraph_numbers
        ):
            add_error(
                f"{OUTPUT_JUDGMENTS} line {line_number}: unavailable citation has paragraph target data"
            )

        chunk_sections = set(record.get("legislation_sections") or [])
        document_sections = set(record.get("document_legislation_sections") or [])
        unknown_chunk_sections = sorted(chunk_sections - valid_sections)
        unknown_document_sections = sorted(document_sections - valid_sections)
        if unknown_chunk_sections:
            add_error(
                f"{OUTPUT_JUDGMENTS} line {line_number}: unknown chunk Act sections {unknown_chunk_sections}"
            )
        if unknown_document_sections:
            add_error(
                f"{OUTPUT_JUDGMENTS} line {line_number}: unknown document Act sections {unknown_document_sections}"
            )

    missing_metadata = sorted(judgment_version_ids - metadata_version_ids)
    metadata_without_chunks = sorted(metadata_version_ids - judgment_version_ids)
    if missing_metadata:
        warnings.append(
            f"{len(missing_metadata)} judgment version_id(s) have chunks but no metadata"
        )
    if metadata_without_chunks:
        warnings.append(
            f"{len(metadata_without_chunks)} metadata record(s) have no chunks"
        )
    missing_case_metadata = sorted(matched_case_version_ids - metadata_version_ids)
    if missing_case_metadata:
        add_error(
            f"{len(missing_case_metadata)} in-baseline case reference(s) point outside metadata"
        )

    return {
        "valid": not errors,
        "errors": errors[:50],
        "error_count": len(errors),
        "warnings": warnings[:50],
        "warning_count": len(warnings),
        **counts,
    }


def process_judgments(
    main_sections: list[str],
) -> tuple[int, int, dict[str, int]]:
    """Write judgment metadata and paragraph-aware retrieval chunks."""
    total_documents = 0
    total_chunks = 0
    stats: Counter[str] = Counter()
    valid_sections = set(main_sections)

    with JUDGMENTS_FILE.open("r", encoding="utf-8") as infile:
        docs = []
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {JUDGMENTS_FILE} line {line_number}"
                ) from error

    baseline_case_index = build_baseline_case_index(docs)

    with (
        OUTPUT_JUDGMENTS.open("w", encoding="utf-8") as chunk_file,
        OUTPUT_JUDGMENT_METADATA.open("w", encoding="utf-8") as metadata_file,
    ):
        for doc in docs:
            total_documents += 1
            parsed = parse_judgment(doc.get("text") or "")
            paragraph_lookup = build_paragraph_lookup(parsed["paragraphs"])
            header = parsed["header"]
            numbering = parsed["status"]
            stats[f"paragraph_numbering_{numbering}"] += 1
            if header["cases_cited"]:
                stats["documents_with_cases_cited"] += 1
            if header["legislation_cited"]:
                stats["documents_with_legislation_cited"] += 1

            (
                document_sections,
                document_parts,
                legislation_references,
            ) = extract_document_act_references(
                header["legislation_cited"], valid_sections, main_sections
            )
            # Raw header citations remain for compatibility; structured
            # references drive retrieval evaluation and display decisions.
            case_references = build_case_references(
                header["cases_cited"], baseline_case_index, paragraph_lookup
            )
            stats["case_references_total"] += len(case_references)
            for reference in case_references:
                stats[f"case_references_{reference['baseline_status']}"] += 1

            metadata = {
                "document_type": "judgment",
                "version_id": doc.get("version_id"),
                "citation": doc.get("citation"),
                "court": doc.get("parsed_court"),
                "date": doc.get("date"),
                "url": doc.get("url"),
                "paragraph_count": len(parsed["paragraphs"]),
                "paragraph_numbering": numbering,
                "paragraph_marker_style": parsed["marker_style"],
                "paragraph_marker_indent": parsed["marker_indent"],
                "catchwords": header["catchwords"],
                "cases_cited": header["cases_cited"],
                "case_references": case_references,
                "legislation_cited": header["legislation_cited"],
                "legislation_references": legislation_references,
                "civil_liability_act_sections": document_sections,
                "civil_liability_act_parts": document_parts,
                "header": header,
            }
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            chunks = make_judgment_chunks(doc, parsed)
            for index, chunk in enumerate(chunks, start=1):
                chunk_sections, chunk_parts = extract_chunk_act_references(
                    chunk["text"],
                    document_sections,
                    valid_sections,
                    main_sections,
                )
                output = {
                    "chunk_id": (
                        f"{doc.get('version_id', 'unknown')}_chunk_{index}"
                    ),
                    "document_type": "judgment",
                    "version_id": doc.get("version_id"),
                    "citation": doc.get("citation"),
                    "court": doc.get("parsed_court"),
                    "date": doc.get("date"),
                    "url": doc.get("url"),
                    "catchwords": header["catchwords"],
                    "paragraph_start": chunk["paragraph_start"],
                    "paragraph_end": chunk["paragraph_end"],
                    "paragraph_numbers": chunk["paragraph_numbers"],
                    "paragraph_numbering": chunk["paragraph_numbering"],
                    "citation_available": chunk["citation_available"],
                    "paragraph_fragment": chunk["paragraph_fragment"],
                    "legislation_sections": chunk_sections,
                    "legislation_parts": chunk_parts,
                    "document_legislation_sections": document_sections,
                    "text": chunk["text"],
                }
                if len(output["text"]) > MAX_CHUNK_CHARS:
                    raise AssertionError(
                        f"Chunk exceeds limit: {output['chunk_id']}"
                    )
                chunk_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                total_chunks += 1

    return total_documents, total_chunks, dict(stats)


def main() -> None:
    print("Processing Civil Liability Act...")
    legislation_provisions, legislation_chunks, main_sections = process_legislation()

    print("Processing judgments...")
    judgment_documents, judgment_chunks, judgment_stats = process_judgments(
        main_sections
    )

    print("Validating processed outputs...")
    validation = validate_processed_outputs(main_sections)

    report = {
        "judgments_processed": judgment_documents,
        "judgment_chunks_created": judgment_chunks,
        "legislation_provisions_created": legislation_provisions,
        "legislation_chunks_created": legislation_chunks,
        "legislation_main_sections_created": len(main_sections),
        "max_chunk_characters": MAX_CHUNK_CHARS,
        "validation": validation,
        **judgment_stats,
    }
    with OUTPUT_REPORT.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print("\nProcessing finished")
    for key, value in report.items():
        print(f"{key}: {value}")
    if not validation["valid"]:
        raise AssertionError(
            f"Processed output validation failed with {validation['error_count']} errors"
        )
    print("\nOutput files:")
    print(OUTPUT_JUDGMENTS)
    print(OUTPUT_JUDGMENT_METADATA)
    print(OUTPUT_LEGISLATION)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
