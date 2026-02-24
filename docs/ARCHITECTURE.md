# 🏗️ Architecture — AutoSuggestion Engine

This document provides a detailed technical breakdown of every component in the system, how they interact, and the design decisions behind them.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Layer Map](#layer-map)
3. [Configuration System](#configuration-system)
4. [Inference Engine](#inference-engine)
5. [API Layer](#api-layer)
6. [Security Architecture](#security-architecture)
7. [Database Layer](#database-layer)
8. [Offline Training Pipeline](#offline-training-pipeline)
9. [ETL Pipeline](#etl-pipeline)
10. [Frontend](#frontend)
11. [Data Formats](#data-formats)
12. [Design Decisions](#design-decisions)

---

## High-Level Overview

```
┌─────────────────────── AUTOSUGGESTION ENGINE ────────────────────────┐
│                                                                       │
│   OFFLINE PIPELINE              RUNTIME ENGINE                        │
│   ──────────────────            ─────────────────────────────────    │
│   Raw CSV Data                  Web Browser / API Client              │
│       │                               │                               │
│   Preprocess                    FastAPI Server (port 8001)            │
│    ├─ char_map                   ├─ /autocomplete/token               │
│    └─ trie + labels              ├─ /autocomplete/suggest             │
│       │                          └─ /autocomplete/feedback            │
│   CNN Training                         │                              │
│    ├─ .keras model               ┌─────┴──────────────────┐           │
│    └─ .tflite INT8               │   Inference Engine     │           │
│                                  │  Trie BFS → CNN rank   │           │
│   ETL Export                     │  + DB feedback merge   │           │
│    └─ feedback → CSVs            └─────────────────────────           │
│                                         │                             │
│   Migration                       MySQL DB                            │
│    └─ schema init                  └─ feedback table                  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Layer Map

```
┌──────────────────────────┐
│         FRONTEND         │   web/frontend/
│  index.html + app.js     │   Static HTML+CSS+JS
│  Debounced keyup → fetch  │   Token stored in memory
└──────────────────────────┘
             │  HTTP REST
             ▼
┌──────────────────────────┐
│    API LAYER (FastAPI)   │   web/backend/autocomplete_api.py
│  Lifespan startup        │   Loads: CSV, Trie, char_map, TFLite, DB
│  3 routes                │   JWT+Fernet auth on 2 of 3 routes
└──────────────────────────┘
         │          │
         ▼          ▼
┌─────────────┐  ┌──────────────────────────────┐
│  Security   │  │   INFERENCE ENGINE            │
│  (core/)    │  │   TrieSearcher  →  TFLite CNN │
│  JWT+Fernet │  │   → Feedback merger           │
└─────────────┘  └──────────────────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │   DATABASE LAYER     │
                  │   mysql connector    │
                  │   pool_size = 5      │
                  │   feedback table     │
                  └──────────────────────┘

┌──────────────────────────┐
│   CONFIGURATION SYSTEM   │   config.py
│   get_config(region,lang)│   Single source of truth
│   Resolves ALL paths      │   Validates region+lang
└──────────────────────────┘
```

---

## Configuration System

### `config.py`

The configuration system uses a **region → language → folder** mapping to navigate the multi-language file hierarchy cleanly.

```
SUPPORTED map:
  "nepal"  → ["nep", "eng"]    # Nepal region supports Nepali + English input
  "india"  → ["hin","ben","tam","tel"]
  "global" → ["eng"]

FOLDER_KEY map:
  "nepal"  → "nep"             # All Nepal files live in .../nep/ folder
  "india"  → None              # India: each language is its own folder
  "global" → "eng"

Resolution: folder = FOLDER_KEY[region] or lang
            suffix = lang           (the *input* language)
```

This separation allows one region (Nepal) to have multiple input languages (English romanization AND Nepali script) all pointing to models trained on the same Nepali output vocabulary.

### Path Resolution Example

```python
cfg = get_config("nepal", "eng")
# Returns:
{
  "TRIE_PATH":    ".../artifacts/nep/trie_eng.json",
  "CHAR_MAP":     ".../artifacts/nep/char_map_eng.json",
  "TFLITE_MODEL": ".../models/nep/model_nep.tflite",   # nep model regardless of input lang
  "META_JSON":    ".../models/nep/meta_nep.json",
  "TRAIN_CSV":    ".../data/train/nep/train_eng.csv",
  "KERAS_FILE":   ".../models/nep/cnn_eng.keras",
  "LABELS":       ".../artifacts/nep/labels_eng.txt",
  "DB_CONFIG":    {...},
  "REGION": "nepal", "LANG": "eng", "FOLDER": "nep"
}
```

---

## Inference Engine

### Step 1: Trie Prefix Search

```python
class TrieNode:
    __slots__ = ("children", "is_word")
    children: dict[str, TrieNode]
    is_word: bool

class TrieSearcher:
    def insert(word) → None          # O(len(word))
    def autocomplete(prefix, limit=500) → list[str]   # BFS
```

**BFS Traversal:**
```
prefix = "na"
Navigate: root → "n" → "a"  (arrive at node for "a")
BFS queue: [(node_a, "na")]
  ├─ node_a.children: {"m": ..., "i": ...}
  ├─ Enqueue: (node_m, "nam"), (node_i, "nai")
  ├─ (node_m).children: {"a": ...}
  │   If is_word: append "na" + accumulated chars
  ... until 500 results or exhausted
```

### Step 2: CNN Scoring

Each candidate is scored against the prefix using the TFLite interpreter:

```python
def _encode(s: str) → np.ndarray:
    arr = [char_map.get(c, 0) for c in s]    # unknown chars → 0
    arr = pad_or_truncate(arr, max_len)        # pad with 0s or truncate
    return np.array(arr, dtype=np.int32)

# For each candidate:
input_tensor = encode(prefix).reshape(1, max_len)
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()
proba = interpreter.get_tensor(output_index)[0]   # shape: (num_classes,)

# INT8 dequantization (if quantized model)
if dtype == np.int8:
    score = scale * (raw_int8 - zero_point)

score = proba[index_of_candidate_in_labels]
```

### Step 3: Feedback Merge

```python
# DB results: lower rank_score = user prefers it more
db_suggestions   = [(label, rank_score), ...]   # rank 1, 2, 3...

# Model results: all get score 999 (high = low priority after sort)
model_ranked     = [(label, 999), ...]

# Merge: DB takes precedence (already in combined dict)
combined = {}
for label, rank in db_suggestions:
    combined[label] = rank
for label, rank in model_ranked:
    if label not in combined:          # DB wins for overlapping labels
        combined[label] = rank

# Sort ascending: lowest rank_score appears first
final = sorted(combined.items(), key=lambda x: x[1])[:top_k]
```

---

## API Layer

### Startup Sequence (`lifespan`)

```
Application startup
    │
    ├── _resolve_config()         Check REGION+LANG env vars → or interactive prompt
    │
    ├── pd.read_csv(TRAIN_CSV)    Load input→target mapping
    │      └─ build eng_to_nep dict
    │
    ├── TrieSearcher.insert(*)    Build trie from all input words
    │
    ├── json.load(CHAR_MAP)       Load character-to-integer mapping
    │
    ├── json.load(META_JSON)      Load max_len
    │
    ├── tf.lite.Interpreter(*)    Load TFLite model + allocate tensors
    │
    └── init_db()                 Ensure feedback table exists
    
Server ready → yield → serve requests
    │
Application shutdown
    └── (no explicit cleanup needed; pool GC'd with process)
```

### Config Resolution Strategy (Non-Interactive Production)

```python
env_region = os.getenv("REGION", "")
env_lang   = os.getenv("LANG",   "")
if env_region and env_lang:
    _CFG = get_config(env_region, env_lang)   # No prompt
else:
    _CFG = prompt_region_and_language()       # Interactive (dev only)
```

This two-mode strategy means the **same code runs in dev and production** without branching.

---

## Security Architecture

### Token Lifecycle

```
                CREATION                         VERIFICATION
─────────────────────────────────    ─────────────────────────────────
payload = {                          Bearer: <fernet_blob>
  "sub": "api_client",                     │
  "exp": now + 120 min                     ▼
}                                    fernet.decrypt(blob)
     │                                     = jwt_string
     ▼                                     │
jwt.encode(payload,                        ▼
  secret, algorithm="HS512")         jwt.decode(jwt_string,
     = jwt_string                      secret, algorithms=["HS512"])
     │                                     │
     ▼                                     ▼
fernet.encrypt(                       payload dict (or raise 401)
  jwt_string.encode()
)  = fernet_blob
     │
     ▼
{"access_token": fernet_blob}
```

### Why Double Encryption?

| Layer | Purpose |
|-------|---------|
| JWT (HS512) | **Integrity** — tamper-proof, contains expiry claim |
| Fernet (AES-128-CBC + HMAC-SHA256) | **Confidentiality** — raw JWT not visible to client; also adds its own HMAC, making forged tokens impossible even without knowing the JWT structure |

---

## Database Layer

### Schema

```sql
feedback
├── input       VARCHAR(255)  -- the typed prefix/query
├── label       VARCHAR(255)  -- the suggestion the user selected
├── rank_score  INT DEFAULT 0 -- incremented on each selection
├── region      VARCHAR(255)  -- e.g. "nepal"
├── lang        VARCHAR(255)  -- e.g. "eng"
└── PRIMARY KEY (input, label)
```

### Connection Pool

```python
_pool = MySQLConnectionPool(
    pool_name="autocomplete_pool",
    pool_size=5,       # 5 simultaneous connections max
    **DB_CONFIG        # host, user, password, database
)
```

On startup, `_ensure_database_exists()` is called first with a temporary pool (no `database` key) to run `CREATE DATABASE IF NOT EXISTS`, then the main pool is created.

### Upsert Pattern

```sql
INSERT INTO feedback (input, label, rank_score)
VALUES (%s, %s, 1)
ON DUPLICATE KEY UPDATE rank_score = rank_score + 1;
```

This is atomic and avoids race conditions when multiple users select the same label simultaneously.

---

## Offline Training Pipeline

### Phase 1: Data Preparation

```
data/csv/kataho_nep_sheet.csv   (raw Nepali dictionary)
     │
     ▼
data/train/nep/train_{lang}.csv  (columns: input, target)
     │
     ├── preprocess/preprocesssing.py
     │     → Reads all unique chars from input+target
     │     → Sorts → assigns index starting from 1
     │     → artifacts/nep/char_map_{lang}.json
     │
     └── preprocess/trie.py
           → Reads correct_word column
           → Inserts each into dict trie
           → artifacts/nep/trie_{lang}.json
           → artifacts/nep/labels_{lang}.txt
```

### Phase 2: Model Training

```bash
python model_trainings/scripts/model_training.py \
  --max_len 12 --emb_dim 32 --epochs 100 --batch 128

# Reads:  TRAIN_CSV, CHAR_MAP, LABELS  (from config)
# Writes: KERAS_FILE (cnn_{lang}.keras)
#         META_JSON  ({"max_len": 12, "emb_dim": 32})
```

**Training data format:**
```csv
input,target
namaste,नमस्ते
namaskar,नमस्कार
...
```

### Phase 3: TFLite Conversion

```bash
python model_trainings/scripts/convert_tflite.py --repr_limit 500

# Reads:  KERAS_FILE, CHAR_MAP, META_JSON, REPR_FILE
# Writes: model_{lang}.tflite  (INT8 quantized)

# INT8 Post-Training Quantization:
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = <500 sample inputs>
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
```

Size reduction: ~1.6 MB Keras → ~155–160 KB TFLite (≈ **10× compression**).

---

## ETL Pipeline

Used to export feedback data for retraining analysis.

```
csv_extractor.py  (orchestrator)
     │
     ├── ETL/utils/extractor.py
     │     query = "SELECT input, label, rank_score, region, lang FROM feedback"
     │     mysql.connector.connect(**DB_CONFIG)
     │     pd.read_sql(query, conn)  → DataFrame
     │
     ├── ETL/utils/transformer.py
     │     df.drop_duplicates()
     │     df.fillna("")
     │
     └── ETL/utils/loader.py
           df.groupby(["region", "lang"])
           → ETL/csv_exports/{region}_{lang}_output.csv
           → UTF-8 BOM encoding (Excel-safe)
```

**Current export files:**
- `ind_hin_output.csv` — India / Hindi
- `nep_eng_output.csv` — Nepal / English input
- `nep_nep_output.csv` — Nepal / Nepali input
- `usa_eng_output.csv` — USA / English

---

## Frontend

The frontend is intentionally **minimal and static** — no framework, no build step.

```
web/frontend/
├── index.html      Single page, loads CSS + JS
├── css/style.css   Responsive card layout, autocomplete dropdown
└── js/app.js       All interaction logic
```

### JS Logic Flow

```javascript
// 1. User types in input#autocompleteInput
input.addEventListener("input", async () => {
    suggestions = await fetchSuggestions(q);  // POST /autocomplete/suggest
    renderSuggestions(suggestions);           // Build <div class="item"> list
});

// 2. User selects (click or Enter key)
function selectWord(word, query) {
    input.value = word;
    clearDropdown();
    fetch(`${API_BASE}/feedback`, { body: {input: query, label: word} });
}

// 3. Keyboard navigation
ArrowDown → activeIndex++
ArrowUp   → activeIndex--
Enter     → items[activeIndex].click()
```

> **Note:** The current `app.js` calls `/suggest` (not `/autocomplete/suggest`) and does not include JWT auth headers. This will need to be updated to match the production API routes.

---

## Data Formats

### `char_map_{lang}.json`

```json
{
  "a": 1, "b": 2, "c": 3, ...,
  "अ": 45, "आ": 46, ...
}
```

Unknown characters map to `0` (padding index).

### `trie_{lang}.json`

```json
{
  "n": {
    "a": {
      "m": {
        "a": {
          "s": {
            "t": {
              "e": { "_end": true }
            }
          }
        }
      }
    }
  }
}
```

### `meta_{lang}.json`

```json
{"max_len": 12, "emb_dim": 32}
```

### `labels_{lang}.txt`

```
namaste
namaskar
nabaras
...
```

One vocabulary word per line. The line index matches the model's output class index.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **TFLite over full TensorFlow** | ~10× smaller model, faster inference, no TF runtime overhead at request time |
| **INT8 quantization** | Further 2-4× size reduction with minimal accuracy loss; calibrated with representative samples |
| **Trie for candidate generation** | O(prefix_len) navigation + BFS is faster and more predictable than fuzzy search for exact-prefix matches |
| **DB rank overrides model** | Recent user choices reflect current preferences better than training data patterns |
| **Double token encryption (JWT + Fernet)** | JWT alone is only integrity-protected (base64-decoded by anyone); Fernet adds confidentiality so clients cannot inspect claims |
| **Connection pool (size=5)** | Avoids per-request connection overhead; `mysql.connector.pooling` handles thread-safety |
| **Deferred `lifespan` config** | Prevents double-prompt when Uvicorn re-imports the module in its worker spawning process |
| **Single `config.py`** | All paths derived at runtime from region+lang → easy to add new languages without changing any other file |
