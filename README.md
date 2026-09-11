# CSEC3888_F14_03_P6

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment

From the project root, create a Python virtual environment:

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run Tests

```bash
pytest tests/
```

### 6. Process the Corpus

Run filtering first, then processing:

```bash
python scripts/filter_corpus.py
python scripts/process_corpus.py
```

`scripts/parse_judgements.py` is imported by `process_corpus.py`; it does not
need to be run separately.

Filtered outputs are written to:

```text
data/filtered/judgments.jsonl
data/filtered/civil_liability_act.json
data/filtered/filter_report.json
```

Processed outputs are written to:

```text
data/processed/judgment_chunks.jsonl
data/processed/judgment_metadata.jsonl
data/processed/legislation_chunks.jsonl
data/processed/processing_report.json
```

### 7. Build the Search Index

Build the saved vector index after processing the corpus:

```bash
python scripts/build_indexes.py
```

This is a one-time operation. It creates the local files under
`data/indexes/` used by the application. The first run downloads the embedding
model.

### 8. Run the Web Application

```bash
streamlit run app.py
```

The first search loads the model and indexes. Later searches in the same
session are substantially faster.

### 9. Deactivate the Virtual Environment

```bash
deactivate
```
