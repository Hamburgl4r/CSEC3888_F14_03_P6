# Corpus Filtering and Processing

This branch contains the dataset filtering and processing work for the project.

## What it does

`filter_corpus.py`
- Filters the Open Australian Legal Corpus
- Keeps NSWSC, NSWCA and NSWDC cases from 2010 onwards
- Keeps cases that mention the Civil Liability Act 2002
- Removes duplicates
- Saves the Civil Liability Act separately

`process_corpus.py`
- Cleans the filtered data
- Splits judgments into smaller chunks
- Keeps metadata such as citation, court, date, URL and paragraph numbers
- Splits the Civil Liability Act into sections
- Prepares the data for retrieval / embeddings

## Corpus download

Download the Open Australian Legal Corpus here:

https://huggingface.co/datasets/isaacus/open-australian-legal-corpus

Create a folder data/ 

Within data/ create folder raw/

Place the downloaded file here:

```
data/raw/corpus.jsonl
````

## Run

From the project root:

```bash
python3 scripts/filter_corpus.py
python3 scripts/process_corpus.py
```

## Why the data files are not on GitHub

The raw corpus and generated filtered/processed files are large, so they are not committed to GitHub.

They can be recreated at any time by downloading the corpus and running the two scripts.

````markdown
## Results

After running:

```bash
python3 scripts/filter_corpus.py
````

the following files will be created in `data/filtered`:

* `civil_liability_act.json` - contains the saved Civil Liability Act 2002 used by the project
* `filter_report.json` - contains a summary of the filtering results, including the number of judgments found for each court
* `judgments.jsonl` - contains the filtered NSW court judgments that mention the Civil Liability Act 2002

After running:

```bash
python3 scripts/process_corpus.py
```

the following files will be created in `data/processed`:

* `judgment_chunks.jsonl` - contains the filtered judgments split into smaller chunks while keeping useful metadata such as citation, court, date, URL and paragraph numbers
* `legislation_chunks.jsonl` - contains the Civil Liability Act split into smaller section-level chunks
* `processing_report.json` - contains a summary of how many judgments, chunks and legislation sections were processed

The processed files are intended to be used in the next stage of the project for retrieval, embeddings and the knowledge base.

```
```
