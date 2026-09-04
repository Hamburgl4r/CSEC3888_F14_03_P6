# This script processes the filtered legal dataset so it is easier to use
# for retrieval and the knowledge base.
#
# It reads the filtered judgments and Civil Liability Act created by
# filter_corpus.py. Judgments are split into smaller paragraph-based chunks
# while keeping useful metadata such as citation, court, date, URL and
# paragraph numbers.
#
# The Civil Liability Act is processed separately and split into sections.
# This makes it easier for the retrieval system to find the most relevant
# judgment paragraphs or legislation sections for a user query.
#
# Outputs:
# - data/processed/judgment_chunks.jsonl
# - data/processed/legislation_chunks.jsonl
# - data/processed/processing_report.json
#
# These processed files are intended to be used later for embeddings,
# vector search and retrieval before relevant evidence is passed to the LLM.

import json
import re
from pathlib import Path


# Input files created by filter_corpus.py
JUDGMENTS_FILE = Path("data/filtered/judgments.jsonl")
ACT_FILE = Path("data/filtered/civil_liability_act.json")

# Folder for processed M3 data
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
OUTPUT_JUDGMENTS = OUTPUT_DIR / "judgment_chunks.jsonl"
OUTPUT_LEGISLATION = OUTPUT_DIR / "legislation_chunks.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "processing_report.json"


# Maximum approximate size of each judgment chunk
# This helps prevent sending entire long judgments into the retrieval system
MAX_CHUNK_CHARS = 3000


def clean_text(text):
    # Return an empty string if there is no text
    if not text:
        return ""

    # Replace repeated spaces and tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove extra whitespace from the start and end
    return text.strip()


def split_into_paragraphs(text):
    # Clean the judgment text before splitting it
    text = clean_text(text)

    if not text:
        return []

    # NSW judgments often use numbered paragraphs such as:
    # [1] First paragraph
    # [2] Second paragraph
    # This pattern splits the text before each numbered paragraph
    pattern = re.compile(r"(?=\[\d+\])")

    parts = pattern.split(text)

    paragraphs = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Try to extract the paragraph number
        match = re.match(r"\[(\d+)\]", part)

        if match:
            paragraph_number = int(match.group(1))
        else:
            # Some text may appear before the first numbered paragraph
            paragraph_number = None

        paragraphs.append({
            "paragraph_number": paragraph_number,
            "text": part
        })

    return paragraphs


def make_judgment_chunks(doc):
    # Get the full judgment text
    text = doc.get("text") or ""

    # Split the judgment into numbered paragraphs
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []

    # Stores the text currently being added to a chunk
    current_text = []

    # Stores paragraph numbers that belong to the current chunk
    current_paragraphs = []

    # Tracks the approximate size of the current chunk
    current_length = 0

    for paragraph in paragraphs:
        paragraph_text = paragraph["text"]
        paragraph_number = paragraph["paragraph_number"]

        paragraph_length = len(paragraph_text)

        # If adding the next paragraph would make the chunk too large,
        # save the current chunk and start a new one
        if (
            current_text
            and current_length + paragraph_length > MAX_CHUNK_CHARS
        ):
            # Use the first and last paragraph numbers for citation metadata
            if current_paragraphs:
                paragraph_start = current_paragraphs[0]
                paragraph_end = current_paragraphs[-1]
            else:
                paragraph_start = None
                paragraph_end = None

            chunks.append({
                "text": "\n".join(current_text),
                "paragraph_start": paragraph_start,
                "paragraph_end": paragraph_end
            })

            # Reset values for the next chunk
            current_text = []
            current_paragraphs = []
            current_length = 0

        # Add this paragraph to the current chunk
        current_text.append(paragraph_text)
        current_length += paragraph_length

        # Only store a paragraph number if one was found
        if paragraph_number is not None:
            current_paragraphs.append(paragraph_number)

    # Save the final chunk after the loop finishes
    if current_text:
        if current_paragraphs:
            paragraph_start = current_paragraphs[0]
            paragraph_end = current_paragraphs[-1]
        else:
            paragraph_start = None
            paragraph_end = None

        chunks.append({
            "text": "\n".join(current_text),
            "paragraph_start": paragraph_start,
            "paragraph_end": paragraph_end
        })

    return chunks


def extract_legislation_references(text):
    # Extract section references found inside judgment chunks
    # Examples:
    # section 5B
    # s 5B
    if not text:
        return []

    references = set()

    pattern = re.compile(
        r"\b(?:section|s)\s+(\d+[A-Za-z]*)",
        re.IGNORECASE
    )

    matches = pattern.findall(text)

    # Use a set first so duplicate section references are removed
    for match in matches:
        references.add(match)

    # Sort the final section numbers to make output easier to read
    return sorted(references)


def process_judgments():
    total_documents = 0
    total_chunks = 0

    # Read each filtered judgment one line at a time
    with JUDGMENTS_FILE.open("r", encoding="utf-8") as infile, \
         OUTPUT_JUDGMENTS.open("w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()

            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON records instead of crashing
                continue

            total_documents += 1

            # Convert one full judgment into smaller paragraph-aware chunks
            chunks = make_judgment_chunks(doc)

            for index, chunk in enumerate(chunks):
                chunk_text = chunk["text"]

                # Keep important metadata with every chunk
                # This allows retrieved results to be traced back to the source
                output = {
                    "chunk_id": (
                        f"{doc.get('version_id', 'unknown')}"
                        f"_chunk_{index + 1}"
                    ),
                    "document_type": "judgment",
                    "version_id": doc.get("version_id"),
                    "citation": doc.get("citation"),
                    "court": doc.get("parsed_court"),
                    "date": doc.get("date"),
                    "url": doc.get("url"),
                    "paragraph_start": chunk["paragraph_start"],
                    "paragraph_end": chunk["paragraph_end"],
                    "legislation_sections": extract_legislation_references(
                        chunk_text
                    ),
                    "text": chunk_text
                }

                # Write each chunk as one JSON line
                outfile.write(
                    json.dumps(output, ensure_ascii=False) + "\n"
                )

                total_chunks += 1

    return total_documents, total_chunks


def split_legislation_into_sections(text):
    # Clean the legislation text before processing it
    text = clean_text(text)

    if not text:
        return []

    # Attempt to identify legislation section headings
    # Examples could look like:
    # 5B General principles
    # 5C Other principles
    # 16 Determination of damages
    section_pattern = re.compile(
        r"(?=^\s*(\d+[A-Za-z]*)\s+[^\n]+)",
        re.MULTILINE
    )

    parts = section_pattern.split(text)

    sections = []

    # Because re.split includes the captured section number,
    # the list contains alternating section numbers and section text
    index = 1

    while index < len(parts):
        section_number = parts[index].strip()

        if index + 1 < len(parts):
            section_text = parts[index + 1].strip()
        else:
            section_text = ""

        if section_text:
            sections.append({
                "section": section_number,
                "text": section_text
            })

        index += 2

    return sections


def process_legislation():
    # Check that the filtered Civil Liability Act exists
    if not ACT_FILE.exists():
        print("Civil Liability Act file was not found")
        return 0

    # Load the saved Act
    with ACT_FILE.open("r", encoding="utf-8") as infile:
        doc = json.load(infile)

    text = doc.get("text") or ""

    # Split the Act into individual sections
    sections = split_legislation_into_sections(text)

    # Save each section as a separate JSON record
    with OUTPUT_LEGISLATION.open("w", encoding="utf-8") as outfile:
        for section in sections:
            output = {
                "chunk_id": (
                    f"{doc.get('version_id', 'civil_liability_act')}"
                    f"_section_{section['section']}"
                ),
                "document_type": "legislation",
                "version_id": doc.get("version_id"),
                "citation": doc.get("citation"),
                "date": doc.get("date"),
                "url": doc.get("url"),
                "section": section["section"],
                "text": section["text"]
            }

            outfile.write(
                json.dumps(output, ensure_ascii=False) + "\n"
            )

    return len(sections)


def main():
    print("Processing judgments...")

    # Process filtered judgments into smaller chunks
    judgment_documents, judgment_chunks = process_judgments()

    print("Processing Civil Liability Act...")

    # Process legislation into section-level chunks
    legislation_sections = process_legislation()

    # Create a summary report so we can verify processing worked
    report = {
        "judgments_processed": judgment_documents,
        "judgment_chunks_created": judgment_chunks,
        "legislation_sections_created": legislation_sections,
        "max_chunk_characters": MAX_CHUNK_CHARS
    }

    with OUTPUT_REPORT.open("w", encoding="utf-8") as report_file:
        json.dump(
            report,
            report_file,
            indent=2
        )

    # Print a simple processing summary
    print("\nProcessing finished")
    print("Judgments processed:", judgment_documents)
    print("Judgment chunks created:", judgment_chunks)
    print("Legislation sections created:", legislation_sections)

    print("\nOutput files:")
    print(OUTPUT_JUDGMENTS)
    print(OUTPUT_LEGISLATION)
    print(OUTPUT_REPORT)


# Only run main() when this file is executed directly
if __name__ == "__main__":
    main()