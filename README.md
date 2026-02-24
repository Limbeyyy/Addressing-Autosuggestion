# 🔤 AutoSuggestion Engine

> **A multilingual, CNN-powered autosuggestion and typo-correction engine with a FastAPI backend, feedback-learning database, and Trie-based prefix search.**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Supported Languages & Regions](#supported-languages--regions)
5. [Component Deep-Dive](#component-deep-dive)
6. [Data Flow](#data-flow)
7. [Quickstart (Local Dev)](#quickstart-local-dev)
8. [Production Deployment](#production-deployment)
9. [Environment Variables](#environment-variables)
10. [API Reference Summary](#api-reference-summary)
11. [Model Architecture](#model-architecture)
12. [Feedback Learning Loop](#feedback-learning-loop)
13. [Project Conventions](#project-conventions)
14. [Documentation Index](#documentation-index)

---

## Overview

The AutoSuggestion Engine is a production-ready, multilingual word-suggestion system. Given a partial input string (prefix), it returns ranked suggestions in the **target script** (e.g., user types English romanization → engine returns Nepali Unicode words).

### ✨ Key Capabilities

| Feature | Detail |
|---|---|
| **Prefix Autocomplete** | Trie-based BFS delivers up to 500 candidate matches instantly |
| **CNN Ranking** | INT8-quantized TFLite CNN model re-ranks candidates by learned probability |
| **Feedback Loop** | User selections stored in MySQL; DB results merge with model results (higher priority) |
| **Multi-language** | Nepal (Nepali/English), India (Hindi/Bengali/Tamil/Telugu), Global (English) |
| **Secure API** | JWT signed with HS512, encrypted with Fernet; token expires in 120 minutes |
| **Production-ready** | Env-var config, connection pooling, CORS lockdown, structured logging |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / CLIENT                                │
│              Web Browser  ·  Mobile App  ·  Any HTTP client         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  HTTPS / REST
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WEB FRONTEND (Static)                           │
│   web/frontend/index.html  ·  css/style.css  ·  js/app.js          │
│   • Keystroke-triggered fetch to /autocomplete/suggest              │
│   • Keyboard navigation (↑↓ Enter)                                  │
│   • On selection → POST /autocomplete/feedback                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  HTTP to port 8001
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND  (autocomplete_api.py)             │
│                                                                     │
│  ┌──────────────┐   ┌────────────────────┐   ┌──────────────────┐  │
│  │  /token      │   │  /suggest          │   │  /feedback       │  │
│  │  Issues      │   │  Trie + CNN        │   │  Upserts rank    │  │
│  │  encrypted   │   │  pipeline          │   │  in MySQL        │  │
│  │  JWT         │   │                    │   │                  │  │
│  └──────────────┘   └────────────────────┘   └──────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Core Security  (core/security.py)                          │    │
│  │  JWT HS512 sign  →  Fernet encrypt  →  Bearer token         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  CORS Middleware  (core/middleware.py)                      │    │
│  │  CORS_ORIGINS env var  →  locked in production              │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────┬────────────────────────┬────────────────────────────┘
               │                        │
               ▼                        ▼
┌──────────────────────┐    ┌───────────────────────────────────┐
│  MySQL               │    │  In-Memory Inference Engine        │
│  autosuggest_db      │    │                                    │
│  ┌────────────────┐  │    │  ┌─────────────────────────────┐  │
│  │ feedback table │  │    │  │  TrieSearcher               │  │
│  │ input | label  │  │    │  │  BFS prefix → 500 candidates│  │
│  │ rank_score     │  │    │  └──────────────┬──────────────┘  │
│  │ region | lang  │  │    │                 │                  │
│  └────────────────┘  │    │  ┌──────────────▼──────────────┐  │
│  Connection Pool     │    │  │  TFLite CNN Interpreter     │  │
│  pool_size=5         │    │  │  INT8 quantized             │  │
└──────────────────────┘    │  │  char_map encode → score    │  │
                            │  └──────────────┬──────────────┘  │
                            │                 │                  │
                            │  ┌──────────────▼──────────────┐  │
                            │  │  Feedback Merger            │  │
                            │  │  DB rank < Model rank(999)  │  │
                            │  │  → final top-K list         │  │
                            │  └─────────────────────────────┘  │
                            └───────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   OFFLINE PIPELINE (one-time setup)                 │
│                                                                     │
│  Raw CSV Data                                                       │
│      │                                                              │
│      ▼                                                              │
│  data_preparation/preprocess/                                       │
│    preprocesssing.py  →  char_map_{lang}.json   (character index)  │
│    trie.py            →  trie_{lang}.json        (prefix trie)     │
│                          labels_{lang}.txt       (vocab list)       │
│      │                                                              │
│      ▼                                                              │
│  model_trainings/scripts/                                           │
│    model_training.py  →  cnn_{lang}.keras        (full Keras model) │
│    convert_tflite.py  →  model_{lang}.tflite     (INT8 quantized)   │
│                                                                     │
│  ETL Pipeline  (csv_extractor.py)                                   │
│    extractor.py  →  transformer.py  →  loader.py                   │
│    MySQL feedback table  →  region/lang CSVs in ETL/csv_exports/   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Autosuggestion_Engine/
│
├── config.py                    # Central config: paths, regions, DB, JWT params
├── autocomplete_main.py         # CLI debug engine (interactive prefix tester)
├── csv_extractor.py             # ETL orchestrator: DB → CSV exports
├── migration.py                 # One-shot DB schema migration runner
├── requirements.txt             # All Python dependencies
├── .env                         # Secrets: JWT_SECRET_KEY, FERNET_KEY, CORS_ORIGINS
├── .gitignore
│
├── web/
│   ├── backend/
│   │   ├── autocomplete_api.py  # FastAPI app — all routes, lifespan, inference
│   │   ├── core/
│   │   │   ├── security.py      # JWT creation/verification + Fernet encryption
│   │   │   └── middleware.py    # CORS middleware setup
│   │   └── database/
│   │       └── db.py            # MySQL connection pool, init_db, update_rank, get_rank
│   └── frontend/
│       ├── index.html           # Single page autocomplete UI
│       ├── css/style.css        # Responsive styles
│       └── js/app.js            # Fetch suggestions, keyboard nav, send feedback
│
├── data_preparation/
│   ├── preprocess/
│   │   ├── preprocesssing.py    # Builds char_map_{lang}.json from CSV
│   │   └── trie.py              # Builds trie_{lang}.json + labels_{lang}.txt
│   ├── artifacts/
│   │   ├── nep/
│   │   │   ├── char_map_eng.json
│   │   │   ├── char_map_nep.json
│   │   │   ├── labels_eng.txt
│   │   │   ├── labels_nep.txt
│   │   │   ├── trie_eng.json    (~37 KB)
│   │   │   └── trie_nep.json    (~42 KB)
│   │   ├── ind/                 # (India language artifacts)
│   │   └── som/                 # (Somali/other artifacts)
│   └── data/
│       ├── csv/
│       │   ├── kataho_nep_sheet.csv        (~200 KB, Nepali dataset)
│       │   └── nep_kataho_code_eng.csv     (~117 KB, English-coded Nepali)
│       ├── train/
│       │   └── nep/
│       │       └── train_{lang}.csv        # Columns: input, target
│       └── samples/
│           └── nep/
│               └── repr_samples_{lang}.txt # Representative samples for INT8 calibration
│
├── model_trainings/
│   ├── scripts/
│   │   ├── model_training.py    # Train CNN on train_{lang}.csv → .keras file
│   │   └── convert_tflite.py    # Convert .keras → INT8 quantized .tflite
│   └── models/
│       └── nep/
│           ├── cnn_eng.keras    (~1.6 MB, full Keras model — English input)
│           ├── cnn_nep.keras    (~1.6 MB, full Keras model — Nepali input)
│           ├── model_eng.tflite (~155 KB, INT8 quantized — English)
│           ├── model_nep.tflite (~157 KB, INT8 quantized — Nepali)
│           ├── meta_eng.json    # {"max_len": N}
│           └── meta_nep.json    # {"max_len": N}
│
├── ETL/
│   ├── utils/
│   │   ├── extractor.py         # Query MySQL feedback → pandas DataFrame
│   │   ├── transformer.py       # Drop duplicates, fill NaN
│   │   └── loader.py            # Group by region/lang → save CSVs
│   └── csv_exports/
│       ├── ind_hin_output.csv
│       ├── nep_eng_output.csv
│       ├── nep_nep_output.csv
│       └── usa_eng_output.csv
│
└── suggest/                     # Python virtual environment (git-ignored)
```

---

## Supported Languages & Regions

| Region | Language Codes | Folder Key | Description |
|--------|----------------|------------|-------------|
| `nepal` | `nep`, `eng` | `nep` | Nepali Unicode output; input can be English romanization or Nepali |
| `india` | `hin`, `ben`, `tam`, `tel` | *(lang code)* | Each Indian language uses its own folder |
| `global` | `eng` | `eng` | General English autocompletion |

> **Folder Key Logic:** The canonical on-disk folder is named after the **target/output language** of the region. For Nepal, all models and artifacts live under `nep/` even when the *input* is English — because the *output* is always Nepali Unicode.

---

## Component Deep-Dive

### 1. `config.py` — Central Configuration

The **single source of truth** for all file paths, region/language definitions, and runtime constants.

```python
# Key exports:
get_config(region, lang)          # → dict of all resolved absolute paths
prompt_region_and_language()      # → interactive CLI selector → calls get_config()

# Constants:
SUPPORTED   = {"nepal": ["nep","eng"], "india": [...], "global": ["eng"]}
FOLDER_KEY  = {"nepal": "nep", "india": None, "global": "eng"}
DB_CONFIG   = {"host":..., "user":..., "password":..., "database":"autosuggest_db"}
TOKEN_ACCESS_TIME = 120   # minutes
ALGORITHM_NAME    = "HS512"
PORT              = 8001
IP_ADDRESS        = "0.0.0.0"
```

**Resolved path dict (example for nepal/eng):**

| Key | Resolved Path |
|-----|---------------|
| `TRIE_PATH` | `data_preparation/artifacts/nep/trie_eng.json` |
| `CHAR_MAP` | `data_preparation/artifacts/nep/char_map_eng.json` |
| `TFLITE_MODEL` | `model_trainings/models/nep/model_nep.tflite` |
| `META_JSON` | `model_trainings/models/nep/meta_nep.json` |
| `TRAIN_CSV` | `data_preparation/data/train/nep/train_eng.csv` |
| `KERAS_FILE` | `model_trainings/models/nep/cnn_eng.keras` |
| `LABELS` | `data_preparation/artifacts/nep/labels_eng.txt` |

---

### 2. Trie Searcher

A **BFS-based prefix trie** built at startup from the training CSV's `input` column.

```
insert("namaste")        → inserts n→a→m→a→s→t→e→{_end}
autocomplete("nam", 500) → BFS from node 'm' → up to 500 words
```

- **API version** (`autocomplete_api.py`): object-oriented `TrieNode` + `TrieSearcher` with `__slots__` for memory efficiency.
- **CLI version** (`autocomplete_main.py`): dict-based trie (simpler, for local testing).

---

### 3. CNN Model & TFLite Inference

**Architecture (`model_training.py`):**
```
Input  (max_len integers)  → int32 sequence
  ↓  Embedding(vocab+1, 32, max_len)
  ↓  Conv1D(128, kernel=3, relu, padding=same)
  ↓  Conv1D(128, kernel=3, relu, padding=same)
  ↓  GlobalMaxPool1D()
  ↓  Dense(128, relu)
  ↓  Dense(num_classes, softmax, float32)
Output  (probability over all vocabulary words)
```

**Training parameters (defaults):**

| Param | Default | Description |
|-------|---------|-------------|
| `--max_len` | 12 | Input sequence length |
| `--emb_dim` | 32 | Embedding dimension |
| `--epochs` | 100 | Max epochs (EarlyStopping patience=10) |
| `--batch` | 128 | Batch size |

**Quantization (`convert_tflite.py`):**
- Full INT8 post-training quantization using representative samples.
- Reduces model size: ~1.6 MB Keras → ~155 KB TFLite.
- Inference uses dequantization formula: `float = scale × (int8 − zero_point)`.

---

### 4. FastAPI Backend (`autocomplete_api.py`)

| Aspect | Detail |
|--------|--------|
| **Startup** | `lifespan` context manager loads CSV, builds Trie, loads char_map, loads TFLite, calls `init_db()` |
| **Config** | `_resolve_config()` checks `REGION`/`LANG` env vars first; falls back to interactive prompt |
| **Port** | `8001` (configurable via `config.py`) |
| **Host** | `0.0.0.0` by default |

**Routes:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/autocomplete/token` | None | Returns an encrypted JWT access token |
| `POST` | `/autocomplete/suggest` | Bearer JWT | Returns top-20 suggestions for a prefix |
| `POST` | `/autocomplete/feedback` | Bearer JWT | Records user's selection in MySQL |

---

### 5. Security Layer (`core/security.py`)

**Double-layer token protection:**
1. **JWT** (HS512): signed with `JWT_SECRET_KEY`, contains `sub="api_client"` + expiry.
2. **Fernet (AES-128-CBC + HMAC)**: JWT string is symmetrically encrypted with `FERNET_KEY` before being sent to clients. Clients send the opaque Fernet-encrypted blob as a Bearer token.

**Flow:**
```
Server                              Client
  │── POST /autocomplete/token ──►  │
  │   create_token({sub:"api_client"})
  │   = jwt.encode(payload, SECRET, HS512)
  │   = fernet.encrypt(jwt_string)
  │◄── {"access_token": "<fernet_blob>"} ──│
  │
  │◄── POST /autocomplete/suggest ──│
  │    Authorization: Bearer <fernet_blob>
  │   fernet.decrypt(blob) → jwt_string
  │   jwt.decode(jwt_string, SECRET) → payload
  │   → verified, proceed
```

---

### 6. Database Layer (`database/db.py`)

```sql
-- Auto-created on startup
CREATE TABLE IF NOT EXISTS feedback (
    input       VARCHAR(255)  NOT NULL,
    label       VARCHAR(255)  NOT NULL,
    rank_score  INT           NOT NULL DEFAULT 0,
    region      VARCHAR(255)  NOT NULL,
    lang        VARCHAR(255)  NOT NULL,
    PRIMARY KEY (input, label)
);
```

| Function | SQL Operation |
|----------|---------------|
| `init_db()` | `CREATE TABLE IF NOT EXISTS feedback` |
| `update_rank(inp, label)` | `INSERT ... ON DUPLICATE KEY UPDATE rank_score = rank_score + 1` |
| `get_rank(label)` | `SELECT SUM(rank_score) FROM feedback WHERE label = ?` |

**Connection pool:** `pool_size=5`, managed by `mysql.connector.pooling.MySQLConnectionPool`.

---

### 7. ETL Pipeline (`csv_extractor.py` + `ETL/`)

Used to export feedback data from MySQL back to CSV for retraining or analysis.

```
csv_extractor.py
    │
    ├── ETL/utils/extractor.py    → SELECT * FROM feedback
    ├── ETL/utils/transformer.py  → drop_duplicates(), fillna("")
    └── ETL/utils/loader.py       → groupby(region, lang) → ETL/csv_exports/{region}_{lang}_output.csv
```

---

## Data Flow

### Inference Request Flow

```
User types prefix "na"
        │
        ▼
  [Frontend JS]  POST /autocomplete/suggest  {text: "na"}
        │  Bearer: <fernet_jwt>
        ▼
  [security.py]  decrypt fernet → decode JWT → verify expiry
        │
        ▼
  [autocomplete_api.py  _score_candidates()]
        ├── TrieSearcher.autocomplete("na", limit=500)  → ["namaste", "nachari", ...]
        │
        ├── For each candidate:
        │      encode("na") → int32 array
        │      TFLite interpreter.invoke()
        │      get output probabilities
        │      score = proba[candidate_index]
        │
        ├── Sort by score DESC → map input→target (eng_to_nep dict) → deduplicate → top-20
        │
        └── DB: SELECT label, rank_score FROM feedback WHERE input LIKE "na%"
                 → merge (DB lower rank_score = higher priority)
        │
        ▼
  {data: [{label: "नमस्ते"}, {label: "नचारी"}, ...]}
        │
        ▼
  [Frontend]  render dropdown → user clicks "नमस्ते"
        │
        ▼
  POST /autocomplete/feedback  {input: "na", label: "नमस्ते"}
        │
        ▼
  [db.py update_rank]  INSERT ... ON DUPLICATE KEY UPDATE rank_score = rank_score + 1
```

---

## Quickstart (Local Dev)

### Prerequisites

- Python 3.10+
- MySQL 8.x running locally
- `pip`

### Steps

```bash
# 1. Clone and enter the project
cd Autosuggestion_Engine

# 2. Create and activate virtual environment
python -m venv suggest
suggest\Scripts\activate          # Windows
# source suggest/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set secrets (copy .env and fill in values)
#    Minimum required: JWT_SECRET_KEY and FERNET_KEY
#    See Environment Variables section below for generation commands

# 5. Run the database migration
python migration.py

# 6. Start the API server (interactive region/lang prompt)
python web\backend\autocomplete_api.py

# 7. Open the frontend
#    Open web/frontend/index.html directly in a browser
#    The JS fetches http://127.0.0.1:8001 by default
```

### CLI Debug Mode

```bash
# Interactive suggestion tester — no server needed
python autocomplete_main.py
# → Prompts for region & language
# → Then: prefix > na
# →   Suggestions: ['नमस्ते', 'नाम', ...]
```

---

## Production Deployment

### Option 1: Direct (Systemd / Process Manager)

```bash
# Set environment variables (do NOT use .env in production)
export JWT_SECRET_KEY="<your-64-char-hex>"
export FERNET_KEY="<your-fernet-base64-url-key>"
export CORS_ORIGINS="https://yourdomain.com"
export REGION="nepal"
export LANG="nep"

# Run with Uvicorn (production grade)
uvicorn web.backend.autocomplete_api:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1 \
  --log-level info
```

> ⚠️ **Set `--workers 1`** because the TFLite interpreter is loaded into process memory at startup. Multi-worker setups require each worker to load its own interpreter (handled correctly by `lifespan`).

### Option 2: Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Secrets injected at runtime, not baked into image
ENV JWT_SECRET_KEY=""
ENV FERNET_KEY=""
ENV CORS_ORIGINS="https://yourdomain.com"
ENV REGION="nepal"
ENV LANG="nep"

EXPOSE 8001

CMD ["uvicorn", "web.backend.autocomplete_api:app", \
     "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

```bash
docker build -t autosuggest .
docker run -d \
  -e JWT_SECRET_KEY="<key>" \
  -e FERNET_KEY="<key>" \
  -e CORS_ORIGINS="https://yourdomain.com" \
  -e REGION="nepal" \
  -e LANG="nep" \
  -p 8001:8001 \
  autosuggest
```

### Option 3: Nginx Reverse Proxy

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Environment Variables

| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET_KEY` | ✅ Yes | `8c8d6e01...` | 32-byte hex string for JWT signing |
| `FERNET_KEY` | ✅ Yes | `yH52MtY4...=` | URL-safe base64 Fernet key for token encryption |
| `CORS_ORIGINS` | ⚠️ Prod | `https://app.com` | Comma-separated allowed origins; `*` for dev |
| `REGION` | ✅ Server | `nepal` | Locks region for non-interactive startup |
| `LANG` | ✅ Server | `nep` | Locks language for non-interactive startup |

**Generate secrets:**
```bash
# JWT secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## API Reference Summary

### POST `/autocomplete/token`
- **Auth:** None
- **Body:** None
- **Response:** `{"access_token": "<fernet_encrypted_jwt>"}`

### POST `/autocomplete/suggest`
- **Auth:** `Authorization: Bearer <token>`
- **Body:** `{"text": "na"}`
- **Response:** `{"data": [{"label": "नमस्ते"}, ...]}`

### POST `/autocomplete/feedback`
- **Auth:** `Authorization: Bearer <token>`
- **Body:** `{"input": "na", "label": "नमस्ते"}`
- **Response:** `{"status": "success"}`

> Full API documentation: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

---

## Model Architecture

```
Input: char-encoded prefix (int32, padded to max_len=12)

Embedding Layer  →  (max_len, 32)
Conv1D (128, k=3, relu, same)  →  (max_len, 128)
Conv1D (128, k=3, relu, same)  →  (max_len, 128)
GlobalMaxPool1D  →  (128,)
Dense (128, relu)  →  (128,)
Dense (num_classes, softmax)   →  (num_classes,)

Loss: sparse_categorical_crossentropy
Optimizer: Adam
EarlyStopping: patience=10, restore_best_weights=True
```

**Files per language variant:**

| File | Size | Purpose |
|------|------|---------|
| `cnn_{lang}.keras` | ~1.6 MB | Full Keras model (training only) |
| `model_{lang}.tflite` | ~155–160 KB | INT8 quantized (inference) |
| `meta_{lang}.json` | 30 B | `{"max_len": N}` |
| `char_map_{lang}.json` | ~0.5–1.1 KB | `{"a": 1, "b": 2, ...}` |
| `labels_{lang}.txt` | ~2–6 KB | Vocabulary words, one per line |
| `trie_{lang}.json` | ~38–43 KB | Serialized prefix trie |

---

## Feedback Learning Loop

```
1. User queries → model returns top-20
2. User clicks suggestion X
3. POST /feedback  {input: prefix, label: X}
4. DB: feedback(input, label).rank_score += 1
5. Next query for same prefix:
   - DB result: (X, rank_score=1) ← lower score = higher priority
   - Model result: (X, 999)       ← model items always get score=999
   → DB wins, X appears first
6. Export: python csv_extractor.py
   → ETL extracts feedback → CSVs in ETL/csv_exports/
   → Can be merged into train CSV for next model retrain cycle
```

---

## Project Conventions

| Convention | Detail |
|------------|--------|
| **Path resolution** | All paths resolved via `config.get_config(region, lang)` — never hardcoded in source files |
| **Config loading** | `sys.path.insert(0, project_root)` pattern used throughout to enable `from config import ...` |
| **Secrets** | Only from env vars (`os.environ.get`); `.env` for local dev via `python-dotenv`, never committed |
| **DB** | Connection pool only; `get_connection()` always called inside `try/finally` to return connections |
| **TFLite** | INT8 dequantization check on every inference call (handles both quantized and float outputs) |
| **Logging** | `logging.basicConfig` at API level; `logging.getLogger(__name__)` in each module |

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — full project overview |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed component architecture with diagrams |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API endpoint documentation |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Step-by-step deployment guide |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | ETL, preprocessing, and model training pipeline |
