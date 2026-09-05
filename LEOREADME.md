# Corpus Filtering and Processing

This branch contains the dataset filtering and processing work for the project.

## What it does

`filter_corpus.py`:

- Filters the Open Australian Legal Corpus.
- Keeps NSWSC, NSWCA and NSWDC cases from 2010 onwards.
- Keeps cases that mention the Civil Liability Act 2002.
- Removes duplicates and saves the Act separately.

`process_corpus.py`:

- Parses all four NSW judgment paragraph formats, including the 2011-2014
  `1Text` style where extraction dropped the separator after the number.
- Reconstructs paragraph numbers lost when HTML ordered lists restart.
- Rejects trailing lists of orders instead of mistaking them for the body.
- Marks judgments without reliable numbering as `citation_available: false`.
- Keeps every retrieval chunk at or below 3,000 characters.
- Extracts catchwords, cited cases and cited legislation into structured data.
- Links chunks only to supported Civil Liability Act sections.
- Splits the Act into sections and schedule clauses with unique IDs.
- Keeps Part, Division and Schedule hierarchy for filtering.

## Corpus download

Download the Open Australian Legal Corpus:

https://huggingface.co/datasets/isaacus/open-australian-legal-corpus

Place the full `corpus.jsonl` file here:

```text
data/raw/corpus.jsonl
```

The project brief pins dataset revision `ef45e3f`. Do not use the partial
Parquet preview shown on the dataset web page.

## Run

From the project root:

```bash
python3 scripts/filter_corpus.py
python3 scripts/process_corpus.py
```

The raw and generated data files are ignored by Git because of their size.
They can be recreated by running the scripts above.

## Outputs

Filtering creates:

- `data/filtered/civil_liability_act.json`
- `data/filtered/filter_report.json`
- `data/filtered/judgments.jsonl`

Processing creates:

- `data/processed/judgment_chunks.jsonl`
- `data/processed/judgment_metadata.jsonl`
- `data/processed/legislation_chunks.jsonl`
- `data/processed/processing_report.json`

## Citation safety

`paragraph_numbering` has one of three values:

- `original`: paragraph markers in the corpus were already continuous.
- `reconstructed`: paragraph order was recovered after the source conversion
  reset visible ordered-list markers.
- `unavailable`: no reliable paragraph sequence was found.

Only chunks with `citation_available: true` may be displayed as pinpoint
paragraph citations. Unavailable chunks can still be searched, but the
application must not invent a paragraph number for them.

Oversized source paragraphs are split into fragments. Every fragment retains
the same paragraph number and includes `paragraph_fragment` metadata.

## Tests

Run:

```bash
python3 -m unittest discover -s tests -v
```
