import json
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/corpus.jsonl")
OUTPUT_DIR = Path("data/filtered")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CASES = OUTPUT_DIR / "judgments.jsonl"
OUTPUT_ACT = OUTPUT_DIR / "civil_liability_act.json"

ACT_ID = "nsw_legislation:2022-06-16/act-2002-022"

VALID_COURTS = {"NSWSC", "NSWCA", "NSWDC"}

citation_pattern = re.compile(
    r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)"
)

total = 0

with INPUT_FILE.open("r", encoding="utf-8") as infile, \
     OUTPUT_CASES.open("w", encoding="utf-8") as outfile:

    for line in infile:

        doc = json.loads(line)

        # Save the Civil Liability Act
        if doc.get("version_id") == ACT_ID:

            with OUTPUT_ACT.open("w", encoding="utf-8") as act_file:
                json.dump(
                    doc,
                    act_file,
                    ensure_ascii=False,
                    indent=2
                )

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

        # Extract the court from the citation
        citation = doc.get("citation") or ""

        matches = citation_pattern.findall(citation)

        if not matches:
            continue

        court = matches[-1][1]

        if court not in VALID_COURTS:
            continue

        doc["parsed_court"] = court

        outfile.write(
            json.dumps(doc, ensure_ascii=False) + "\n"
        )

        total += 1


print("Filtering finished")
print("Total judgments:", total)