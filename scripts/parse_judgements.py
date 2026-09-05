"""Parse the semi-structured headers and paragraphs in NSW judgments.

The plain-text corpus contains four paragraph-marker styles:

* ``1. Text`` (modern judgments)
* ``  1 Text`` (older judgments)
* ``[1] Text`` (a small number of judgments)
* ``1Text`` (2011-2014 judgments, where extraction dropped the separator)
* ``1 Text`` and ``1.Text`` (legacy judgments with no marker indentation)

The HTML-to-text conversion also loses the ``start`` value of some ordered
lists.  In those documents the visible marker restarts at 1 after a heading,
even though the source paragraph numbering continues.  We reconstruct those
numbers only when a consistent top-level marker style can be identified.
Documents without a reliable marker sequence are deliberately marked
``unavailable`` instead of receiving made-up pinpoint citations.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


LABEL_ALIASES = {
    "medium neutral citation": "medium_neutral_citation",
    "citation": "medium_neutral_citation",
    "hearing dates": "hearing_dates",
    "hearing date(s)": "hearing_dates",
    "decision date": "decision_date",
    "judgment date": "decision_date",
    "date of orders": "date_of_orders",
    "jurisdiction": "jurisdiction",
    "before": "before",
    "judgment of": "before",
    "decision": "decision",
    "catchwords": "catchwords",
    "legislation cited": "legislation_cited",
    "cases cited": "cases_cited",
    "parties": "parties",
    "file number(s)": "file_numbers",
    "file numbers": "file_numbers",
    "counsel": "counsel",
    "solicitors": "solicitors",
}

HEADER_LABEL_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 ()/'&.-]{1,60}):\s*(.*)$"
)
BODY_HEADING_RE = re.compile(
    r"^[ \t]*(?:JUDGMENT|JUDGMENTS|REASONS FOR JUDGMENT|REASONS FOR DECISION)[ \t]*$",
    re.I | re.M,
)

MARKER_PATTERNS = {
    "bracket": re.compile(r"^([ \t]*)\[(\d{1,4})\][ \t]+"),
    "dot": re.compile(r"^([ \t]*)(\d{1,4})\.[ \t]+"),
    "dot_glued": re.compile(r"^()(\d{1,4})\.(?=[A-Z])"),
    # Requiring indentation prevents years and ordinary sentence text from
    # being mistaken for the older ``  1 Text`` paragraph style.
    "plain": re.compile(r"^([ \t]+)(\d{1,4})[ \t]+"),
    # Some legacy NSW exports use unindented plain markers. Restricting the
    # following character to uppercase keeps ordinary numeric lists out.
    "plain_unindented": re.compile(r"^()(\d{1,4})[ \t]+(?=[A-Z])"),
    # ``8The respondent said`` - the separator between the paragraph number
    # and the text was lost when the corpus was extracted from HTML.
    "glued": re.compile(r"^()(\d{1,4})(?=[A-Z])"),
}

# A marker group starting this late in a document is almost always the list of
# orders at the foot of a judgment rather than the numbered body.
MAX_BODY_START_FRACTION = 0.6
LARGE_GROUP_MARKERS = 20

# If one "paragraph" swallows most of the body and is far longer than any real
# paragraph, the markers belonged to a short list (usually the orders) and the
# body itself is unnumbered. Genuine paragraphs quoting legislation at length
# stay well below the character limit.
MAX_SINGLE_PARAGRAPH_FRACTION = 0.5
MAX_SINGLE_PARAGRAPH_CHARS = 20000


def _normalise_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _indent_width(value: str) -> int:
    return len(value.expandtabs(4))


def _clean_block(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _marker_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    offset = 0

    for line_number, line_with_ending in enumerate(
        text.splitlines(keepends=True), start=1
    ):
        line = line_with_ending.rstrip("\r\n")
        for style, pattern in MARKER_PATTERNS.items():
            match = pattern.match(line)
            if match:
                candidates.append(
                    {
                        "style": style,
                        "indent": _indent_width(match.group(1)),
                        "number": int(match.group(2)),
                        "start": offset,
                        "content_start": offset + match.end(),
                        "line_number": line_number,
                    }
                )
                break
        offset += len(line_with_ending)

    return candidates


def _sequence_quality(numbers: list[int]) -> float:
    if len(numbers) < 2:
        return 0.0
    valid_steps = sum(
        current == previous + 1 or current == 1
        for previous, current in zip(numbers, numbers[1:])
    )
    return valid_steps / (len(numbers) - 1)


def _trim_before_body_heading(
    markers: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    """Discard header/table markers when a real body sequence follows."""
    if not markers:
        return markers

    heading_ends = []
    for match in BODY_HEADING_RE.finditer(text):
        markers_before_heading = sum(
            marker["start"] < match.start() for marker in markers
        )
        if markers_before_heading <= 1 and match.start() < markers[-1]["start"]:
            heading_ends.append(match.end())
    if not heading_ends:
        return markers

    for heading_end in reversed(heading_ends):
        after_heading = [
            marker for marker in markers if marker["start"] >= heading_end
        ]
        if len(after_heading) < 3:
            continue
        numbers = [marker["number"] for marker in after_heading]
        if numbers[0] <= 3 and _sequence_quality(numbers) >= 0.55:
            return after_heading

    return markers


def _select_cross_indent_sequence(
    candidates: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    """Recover legacy judgments whose body markers change indentation."""
    body_heading = None
    for match in BODY_HEADING_RE.finditer(text):
        body_heading = match.end()
        break

    text_length = len(text)
    allowed_styles = {"plain", "plain_unindented", "dot", "dot_glued", "glued"}
    body_candidates = [
        candidate
        for candidate in candidates
        if candidate["style"] in allowed_styles
        and (body_heading is None or candidate["start"] >= body_heading)
    ]

    best: list[dict[str, Any]] = []
    for start_index, start in enumerate(body_candidates):
        if start["number"] > 3:
            continue
        if start["start"] / max(1, text_length) > MAX_BODY_START_FRACTION:
            continue
        sequence = [start]
        expected = start["number"] + 1
        for candidate in body_candidates[start_index + 1:]:
            if candidate["number"] == expected:
                sequence.append(candidate)
                expected += 1
        if len(sequence) > len(best):
            best = sequence

    if len(best) < 10:
        return []
    return best


def _select_marker_group(
    candidates: list[dict[str, Any]],
    text: str,
) -> list[dict[str, Any]]:
    text_length = len(text)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate["style"], candidate["indent"])].append(candidate)

    ranked: list[tuple[float, list[dict[str, Any]]]] = []
    style_bonus = {
        "dot": 1.08,
        "plain": 1.04,
        "glued": 1.04,
        "bracket": 1.0,
        "dot_glued": 1.02,
        "plain_unindented": 0.98,
    }

    for (style, indent), group in groups.items():
        numbers = [item["number"] for item in group]
        if len(group) < 3:
            continue

        quality = _sequence_quality(numbers)
        starts_like_paragraphs = numbers[0] <= 3
        # A judgment excerpt can begin above paragraph 3, but in that case it
        # must provide a long, strongly monotonic sequence.
        if not starts_like_paragraphs and not (
            len(group) >= 10 and quality >= 0.8
        ):
            continue
        if quality < 0.55:
            continue

        body_start_fraction = group[0]["start"] / max(1, text_length)
        if (
            body_start_fraction > MAX_BODY_START_FRACTION
            and len(group) < LARGE_GROUP_MARKERS
        ):
            continue

        score = (
            len(group)
            * (0.5 + quality)
            * style_bonus[style]
            * (1 - body_start_fraction)
            / (1 + indent * 0.015)
        )
        if numbers[0] == 1:
            score *= 1.1
        ranked.append((score, group))

    if not ranked:
        return []

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[0][1]

    # Ignore unrelated numeric material before the first plausible paragraph,
    # but only when it is a short preamble. Trimming a long run would discard
    # most of the judgment.
    first_start = next(
        (
            index
            for index, marker in enumerate(selected)
            if marker["number"] <= 3
        ),
        0,
    )
    if first_start > len(selected) // 4:
        first_start = 0
    selected = _trim_before_body_heading(selected[first_start:], text)

    cross_indent = _select_cross_indent_sequence(candidates, text)
    if (
        len(cross_indent) > len(selected)
        and cross_indent[0]["start"] <= selected[0]["start"]
    ):
        return cross_indent

    return selected


def parse_paragraphs(text: str) -> dict[str, Any]:
    """Return parsed paragraphs and paragraph-numbering diagnostics."""
    if not text:
        return {
            "paragraphs": [],
            "status": "unavailable",
            "body_start": None,
            "marker_style": None,
            "marker_indent": None,
        }

    markers = _select_marker_group(_marker_candidates(text), text)
    if not markers:
        return {
            "paragraphs": [],
            "status": "unavailable",
            "body_start": None,
            "marker_style": None,
            "marker_indent": None,
        }

    displayed = [marker["number"] for marker in markers]
    is_original = all(
        current == previous + 1
        for previous, current in zip(displayed, displayed[1:])
    )
    status = "original" if is_original else "reconstructed"
    first_number = displayed[0]

    paragraphs = []
    for index, marker in enumerate(markers):
        end = markers[index + 1]["start"] if index + 1 < len(markers) else len(text)
        assigned_number = (
            marker["number"] if is_original else first_number + index
        )
        content = _clean_block(text[marker["content_start"]:end])
        if not content:
            continue
        paragraphs.append(
            {
                "paragraph_number": assigned_number,
                "displayed_number": marker["number"],
                "text": f"[{assigned_number}] {content}",
                "source_line": marker["line_number"],
            }
        )

    if paragraphs:
        body_length = sum(len(item["text"]) for item in paragraphs)
        longest = max(len(item["text"]) for item in paragraphs)
        if (
            longest > MAX_SINGLE_PARAGRAPH_CHARS
            and longest > MAX_SINGLE_PARAGRAPH_FRACTION * body_length
        ):
            paragraphs = []

    return {
        "paragraphs": paragraphs,
        "status": status if paragraphs else "unavailable",
        "body_start": markers[0]["start"] if paragraphs else None,
        "marker_style": markers[0]["style"] if paragraphs else None,
        "marker_indent": markers[0]["indent"] if paragraphs else None,
    }


def parse_header(text: str, body_start: int | None = None) -> dict[str, Any]:
    """Extract labelled header fields from the text before the judgment body."""
    header_text = text if body_start is None else text[:body_start]
    fields: dict[str, list[str]] = defaultdict(list)
    current_field: str | None = None

    for raw_line in header_text.splitlines():
        stripped = raw_line.strip()
        if stripped.upper() in {
            "JUDGMENT",
            "JUDGMENTS",
            "REASONS FOR JUDGMENT",
            "REASONS FOR DECISION",
        }:
            current_field = None
            continue

        match = HEADER_LABEL_RE.match(raw_line)
        if match:
            label = _normalise_label(match.group(1))
            current_field = LABEL_ALIASES.get(label)
            if current_field and match.group(2).strip():
                fields[current_field].append(match.group(2).strip())
            continue

        if current_field and stripped:
            # Header continuations are indented in both the modern and legacy
            # layouts. Unindented prose is not part of a header field.
            if raw_line[:1].isspace():
                fields[current_field].append(stripped)
            else:
                current_field = None

    # In the legacy two-column layout, the first case can appear one line
    # above the ``CASES CITED:`` label. It is then initially read as a
    # continuation of ``LEGISLATION CITED:``. Move obvious case names back.
    if "cases_cited" in fields:
        misplaced_cases = []
        legislation_values = fields.get("legislation_cited", [])
        while legislation_values and re.search(
            r"\b(?:v|versus)\b|\bRe\s+\S", legislation_values[-1], re.I
        ):
            misplaced_cases.insert(0, legislation_values.pop())
        if misplaced_cases:
            fields["cases_cited"] = misplaced_cases + fields["cases_cited"]

    scalar_fields = {
        "medium_neutral_citation",
        "hearing_dates",
        "decision_date",
        "date_of_orders",
        "jurisdiction",
        "before",
        "decision",
        "catchwords",
        "parties",
        "file_numbers",
        "counsel",
        "solicitors",
    }
    result: dict[str, Any] = {}
    for name, values in fields.items():
        cleaned_values = [value for value in values if value]
        if name in scalar_fields:
            result[name] = "\n".join(cleaned_values)
        else:
            result[name] = cleaned_values

    result.setdefault("cases_cited", [])
    result.setdefault("legislation_cited", [])
    result.setdefault("catchwords", "")
    return result


def parse_judgment(text: str) -> dict[str, Any]:
    paragraph_data = parse_paragraphs(text)
    return {
        "header": parse_header(text, paragraph_data["body_start"]),
        **paragraph_data,
    }
