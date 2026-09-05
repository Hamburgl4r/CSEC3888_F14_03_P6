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

### 7. Deactivate the Virtual Environment

```bash
deactivate
```
