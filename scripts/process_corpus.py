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
ACT_NAME_RE = re.compile(r"\bCivil\s+Liability\s+Act(?:\s+2002)?\b", re.I)
SECTION_REFERENCE_RE = re.compile(
    # ``s 5D``, ``ss 5B-5D`` and the unspaced ``s5D`` all occur.
    r"\b(?:ss?|sections?)\s*"
    r"(\d+[A-Za-z]*(?:\([^)\s]+\))*"
    r"(?:\s*(?:,|and|to|[-–—])\s*"
    r"\d+[A-Za-z]*(?:\([^)\s]+\))*)*)",
    re.I,
)
PART_REFERENCE_RE = re.compile(r"\b(?:Pt|Part)\s+(\d+[A-Za-z]*)\b", re.I)
# A subsection number in ``5D(1)(a)`` is not another section reference.
SECTION_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9(])\d+[A-Za-z]*")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


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

        is_range = bool(re.search(r"\bto\b|[-–—]", expression, re.I))
        if is_range and len(tokens) >= 2:
            expanded = _expand_section_range(tokens[0], tokens[-1], section_order)
            tokens = expanded + tokens[1:-1]

        for token in tokens:
            base = re.match(r"\d+[A-Z]*", token).group(0)
            if base in valid_lookup:
                references.add(valid_lookup[base])

    order_lookup = {section: index for index, section in enumerate(section_order)}
    return sorted(references, key=lambda section: order_lookup.get(section, 10**9))


def extract_document_act_references(
    legislation_cited: list[str],
    valid_sections: set[str],
    section_order: list[str],
) -> tuple[list[str], list[str]]:
    sections: set[str] = set()
    parts: set[str] = set()
    for item in legislation_cited:
        if not ACT_NAME_RE.search(item):
            continue
        sections.update(extract_section_references(item, valid_sections, section_order))
        parts.update(PART_REFERENCE_RE.findall(item))
    return (
        [section for section in section_order if section in sections],
        sorted(parts),
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
    sections = set(
        extract_section_references(text, set(document_sections), section_order)
    )
    parts: set[str] = set()

    for act_match in ACT_NAME_RE.finditer(text):
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
            extract_section_references(window, valid_sections, section_order)
        )
        parts.update(PART_REFERENCE_RE.findall(window))

    return (
        [section for section in section_order if section in sections],
        sorted(parts),
    )


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


def process_legislation() -> tuple[int, list[str]]:
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

    with OUTPUT_LEGISLATION.open("w", encoding="utf-8") as outfile:
        for provision in provisions:
            output = {
                "chunk_id": (
                    f"{doc.get('version_id', 'civil_liability_act')}_"
                    f"{provision['provision_id']}"
                ),
                "document_type": "legislation",
                "version_id": doc.get("version_id"),
                "citation": doc.get("citation"),
                "date": doc.get("date"),
                "url": doc.get("url"),
                **provision,
            }
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")

    return len(provisions), main_sections


def process_judgments(
    main_sections: list[str],
) -> tuple[int, int, dict[str, int]]:
    total_documents = 0
    total_chunks = 0
    stats: Counter[str] = Counter()
    valid_sections = set(main_sections)

    with (
        JUDGMENTS_FILE.open("r", encoding="utf-8") as infile,
        OUTPUT_JUDGMENTS.open("w", encoding="utf-8") as chunk_file,
        OUTPUT_JUDGMENT_METADATA.open("w", encoding="utf-8") as metadata_file,
    ):
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {JUDGMENTS_FILE} line {line_number}"
                ) from error

            total_documents += 1
            parsed = parse_judgment(doc.get("text") or "")
            header = parsed["header"]
            numbering = parsed["status"]
            stats[f"paragraph_numbering_{numbering}"] += 1
            if header["cases_cited"]:
                stats["documents_with_cases_cited"] += 1
            if header["legislation_cited"]:
                stats["documents_with_legislation_cited"] += 1

            document_sections, document_parts = extract_document_act_references(
                header["legislation_cited"], valid_sections, main_sections
            )
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
                "legislation_cited": header["legislation_cited"],
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
    legislation_provisions, main_sections = process_legislation()

    print("Processing judgments...")
    judgment_documents, judgment_chunks, judgment_stats = process_judgments(
        main_sections
    )

    report = {
        "judgments_processed": judgment_documents,
        "judgment_chunks_created": judgment_chunks,
        "legislation_provisions_created": legislation_provisions,
        "legislation_main_sections_created": len(main_sections),
        "max_chunk_characters": MAX_CHUNK_CHARS,
        **judgment_stats,
    }
    with OUTPUT_REPORT.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print("\nProcessing finished")
    for key, value in report.items():
        print(f"{key}: {value}")
    print("\nOutput files:")
    print(OUTPUT_JUDGMENTS)
    print(OUTPUT_JUDGMENT_METADATA)
    print(OUTPUT_LEGISLATION)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
