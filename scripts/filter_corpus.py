# This script filters the raw Open Australian Legal Corpus for the project.
# It keeps NSW court decisions from 2010 onwards that mention the
# Civil Liability Act 2002 and are from NSWSC, NSWCA or NSWDC.
#
# It also removes duplicate judgments using version_id and saves a copy
# of the Civil Liability Act so it can be processed separately later.
#
# Outputs:
# - data/filtered/judgments.jsonl
# - data/filtered/civil_liability_act.json
# - data/filtered/filter_report.json
#
# The main purpose of this script is to reduce the large raw corpus into
# a smaller, relevant dataset that can be used by the next processing stage.

import json
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/corpus.jsonl")
OUTPUT_DIR = Path("data/filtered")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CASES = OUTPUT_DIR / "judgments.jsonl"
OUTPUT_ACT = OUTPUT_DIR / "civil_liability_act.json"
OUTPUT_REPORT = OUTPUT_DIR / "filter_report.json"

ACT_ID = "nsw_legislation:2022-06-16/act-2002-022"

VALID_COURTS = {"NSWSC", "NSWCA", "NSWDC"}

citation_pattern = re.compile(
    r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)"
)

seen_version_ids = set()

court_counts = {
    "NSWSC": 0,
    "NSWCA": 0,
    "NSWDC": 0
}

total = 0
duplicates = 0
invalid_json = 0
cases_cited_count = 0
act_found = False


with INPUT_FILE.open("r", encoding="utf-8") as infile, \
     OUTPUT_CASES.open("w", encoding="utf-8") as outfile:

    for line in infile:

        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            invalid_json += 1
            continue

        version_id = doc.get("version_id")

        # Save the Civil Liability Act
        if version_id == ACT_ID:
            with OUTPUT_ACT.open("w", encoding="utf-8") as act_file:
                json.dump(
                    doc,
                    act_file,
                    ensure_ascii=False,
                    indent=2
                )

            act_found = True

        # Only keep court decisions
        if doc.get("type") != "decision":
            continue

        # Must be from 2010 onwards
        date = doc.get("date")

        if not date or date < "2010-01-01":
            continue

        # Must mention Civil Liability Act 2002
        text = doc.get("text") or ""

        if "civil liability act 2002" not in text.lower():
            continue

        # Extract court from citation
        citation = doc.get("citation") or ""

        matches = citation_pattern.findall(citation)

        if not matches:
            continue

        # Use the final citation match
        court = matches[-1][1]

        if court not in VALID_COURTS:
            continue

        # Deduplicate using version_id
        if not version_id:
            continue

        if version_id in seen_version_ids:
            duplicates += 1
            continue

        seen_version_ids.add(version_id)

        # Add parsed court field
        doc["parsed_court"] = court

        outfile.write(
            json.dumps(doc, ensure_ascii=False) + "\n"
        )

        total += 1
        court_counts[court] += 1

        if "cases cited:" in text.lower():
            cases_cited_count += 1


# Calculate percentage
if total > 0:
    cases_cited_percentage = (
        cases_cited_count / total
    ) * 100
else:
    cases_cited_percentage = 0


report = {
    "act_found": act_found,
    "total_judgments": total,
    "court_counts": court_counts,
    "cases_cited_count": cases_cited_count,
    "cases_cited_percentage": round(cases_cited_percentage, 2),
    "duplicates_removed": duplicates,
    "invalid_json_lines": invalid_json
}


with OUTPUT_REPORT.open("w", encoding="utf-8") as report_file:
    json.dump(
        report,
        report_file,
        indent=2
    )


print("Filtering finished")
print("Act found:", act_found)
print("Total judgments:", total)

print("\nCourt breakdown:")
print("NSWSC:", court_counts["NSWSC"])
print("NSWCA:", court_counts["NSWCA"])
print("NSWDC:", court_counts["NSWDC"])

print(
    "\nCases Cited:",
    cases_cited_count,
    f"({cases_cited_percentage:.2f}%)"
)

print("Duplicates removed:", duplicates)
print("Invalid JSON lines:", invalid_json)