# 🔄 Data Pipeline Guide — AutoSuggestion Engine

This document covers the complete lifecycle of data in the system: from raw source CSV files through preprocessing, model training, TFLite export, and the feedback ETL loop.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Training Data Format](#training-data-format)
3. [Phase 1: Preprocessing](#phase-1-preprocessing)
4. [Phase 2: Artifact Generation](#phase-2-artifact-generation)
5. [Phase 3: Model Training](#phase-3-model-training)
6. [Phase 4: TFLite Conversion](#phase-4-tflite-conversion)
7. [Phase 5: Feedback ETL](#phase-5-feedback-etl)
8. [Adding a New Language](#adding-a-new-language)
9. [File Reference](#file-reference)

---

## Pipeline Overview

```
Raw CSV Data
  (data/csv/)
      │
      ▼  Manual curation → structured train CSV
Training CSV
  (data/train/{folder}/train_{lang}.csv)
      │
      ├─── Phase 1: Preprocessing ──────────────────────────────────
      │    preprocess/preprocesssing.py
      │      → reads all chars from input + target columns
      │      → builds char → integer index (from 1)
      │      → artifacts/{folder}/char_map_{lang}.json
      │
      ├─── Phase 2: Trie + Labels ──────────────────────────────────
      │    preprocess/trie.py
      │      → reads vocabulary from training CSV
      │      → builds prefix trie (nested dict)
      │      → artifacts/{folder}/trie_{lang}.json
      │      → artifacts/{folder}/labels_{lang}.txt
      │
      ├─── Phase 3: CNN Training ───────────────────────────────────
      │    model_trainings/scripts/model_training.py
      │      → encodes input sequences using char_map
      │      → trains Embedding+Conv1D+Dense CNN
      │      → models/{folder}/cnn_{lang}.keras
      │      → models/{folder}/meta_{lang}.json
      │
      └─── Phase 4: TFLite Export ──────────────────────────────────
           model_trainings/scripts/convert_tflite.py
             → loads .keras model
             → INT8 post-training quantization
             → calibrated with representative samples
             → models/{folder}/model_{folder}.tflite

Runtime Feedback Loop:
  User selections → MySQL feedback table
      │
      ▼  Phase 5: ETL Export
  csv_extractor.py
    ETL/utils/extractor.py  →  transformer.py  →  loader.py
    → ETL/csv_exports/{region}_{lang}_output.csv
    → (optionally merge back into train CSV for next cycle)
```

---

## Training Data Format

All training CSV files follow this schema:

```csv
input,target
namaste,नमस्ते
namaskar,नमस्कार
naam,नाम
raamro,राम्रो
```

| Column | Type | Description |
|--------|------|-------------|
| `input` | string | The romanized/input language form (lowercased) |
| `target` | string | The output word in the target script (e.g., Nepali Unicode) |

### File Locations

```
data_preparation/data/train/
└── nep/
    ├── train_eng.csv     ← English romanization → Nepali
    └── train_nep.csv     ← Nepali script → Nepali

data_preparation/data/csv/
├── kataho_nep_sheet.csv       ← Raw Nepali word list (~200 KB)
└── nep_kataho_code_eng.csv    ← English-coded Nepali words (~117 KB)
```

---

## Phase 1: Preprocessing

**Script:** `data_preparation/preprocess/preprocesssing.py`

```bash
python data_preparation/preprocess/preprocesssing.py \
  --train_csv data_preparation/data/train/nep/train_eng.csv
```

**What it does:**

```python
# 1. Load CSV
df = pd.read_csv(train_csv)

# 2. Collect ALL unique characters from both columns
all_samples = list(df["input"]) + list(df["target"])
chars = sorted(set(char for s in all_samples for char in s))

# 3. Assign integer index (1-based; 0 is padding)
char_map = {c: i+1 for i, c in enumerate(chars)}

# 4. Save
json.dump(char_map, f)
```

**Output:** `data_preparation/artifacts/{folder}/char_map_{lang}.json`

```json
{
  " ": 1,
  "a": 2,
  "b": 3,
  ...
  "अ": 54,
  "आ": 55
}
```

**Why 1-based indexing?**  
Index `0` is reserved as the **padding token**. Short sequences are padded with zeros to reach `max_len`, and the embedding layer uses index 0 for the padding embedding.

---

## Phase 2: Artifact Generation

**Script:** `data_preparation/preprocess/trie.py`

```bash
python data_preparation/preprocess/trie.py
```

**What it does:**

```python
# 1. Read vocabulary
df = pd.read_csv(TRAIN_CSV)
vocab = df['correct_word'].dropna().str.strip().tolist()

# 2. Build nested dict trie
root = {}
for word in vocab:
    node = root
    for ch in word:
        node = node.setdefault(ch, {})
    node['_end'] = True

# 3. Save trie
json.dump(root, trie_file)

# 4. Save labels (one per line, order matters — must match training labels)
for w in vocab:
    labels_file.write(w + "\n")
```

**Outputs:**
- `artifacts/{folder}/trie_{lang}.json` — serialized prefix search structure
- `artifacts/{folder}/labels_{lang}.txt` — vocabulary list; **line N** = **class index N** for the CNN

> ⚠️ **Critical:** The order of words in `labels_{lang}.txt` must match the order used during model training (`label2i` mapping). Never re-sort this file after training.

---

## Phase 3: Model Training

**Script:** `model_trainings/scripts/model_training.py`

```bash
python model_trainings/scripts/model_training.py \
  --max_len 12 \
  --emb_dim 32 \
  --epochs  100 \
  --batch   128
```

**Full Training Flow:**

```python
# 1. Load data
df = pd.read_csv(TRAIN_CSV)
char2i = json.load(open(CHAR_MAP))
labels = open(LABELS).read().splitlines()
label2i = {label: i for i, label in enumerate(labels)}

# 2. Encode inputs: each word → int32 array of length max_len
X = np.array([encode_text(s, char2i, max_len) for s in df['input']])
y = np.array([label2i[t] for t in df['target']])

# 3. Build model
inp = Input(shape=(max_len,), dtype='int32')
x   = Embedding(vocab_size+1, emb_dim, input_length=max_len)(inp)
x   = Conv1D(128, 3, activation='relu', padding='same')(x)
x   = Conv1D(128, 3, activation='relu', padding='same')(x)
x   = GlobalMaxPool1D()(x)
x   = Dense(128, activation='relu')(x)
out = Dense(num_classes, activation='softmax', dtype='float32')(x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train with early stopping
model.fit(X, y,
          epochs=epochs,
          batch_size=batch,
          validation_split=0.1,
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

# 5. Save
model.save(KERAS_FILE, include_optimizer=False)
json.dump({"max_len": max_len, "emb_dim": emb_dim}, meta_file)
```

**Model Architecture Summary:**

```
Input: (batch, 12)         int32 sequence of char indices
  ↓
Embedding: (batch, 12, 32) lookup table (vocab_size+1 → 32-dim vector)
  ↓
Conv1D ×2: (batch, 12, 128) captures character n-gram patterns
  ↓
GlobalMaxPool1D: (batch, 128) single most-activated feature per filter
  ↓
Dense(128, relu): (batch, 128) non-linear combination
  ↓
Dense(N, softmax): (batch, N) probability over all vocabulary words
```

**Hyperparameter Rationale:**

| Param | Why this value |
|-------|----------------|
| `max_len=12` | Most romanized words ≤12 characters; covers edge cases |
| `emb_dim=32` | Small enough for TFLite; large enough for character relationships |
| `Conv1D(128, 3)` | k=3 captures trigram-level patterns (common in romanization) |
| `GlobalMaxPool` | Extracts most significant pattern; invariant to position |
| `patience=10` | Allows model to escape local minima while preventing overfitting |

---

## Phase 4: TFLite Conversion

**Script:** `model_trainings/scripts/convert_tflite.py`

**Requires:** A representative samples file:
```
data_preparation/data/samples/{folder}/repr_samples_{lang}.txt
```

This is a plain text file with one input word per line, representing typical inputs for calibration.

```bash
python model_trainings/scripts/convert_tflite.py --repr_limit 500
```

**Conversion Process:**

```python
# 1. Load Keras model
model = tf.keras.models.load_model(KERAS_FILE)

# 2. Load calibration samples
repr_lines = open(REPR_FILE).read().splitlines()[:500]
repr_data  = [encode_line(s, char_map, max_len) for s in repr_lines]

# 3. Configure INT8 quantized converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: (arr.reshape(1,-1) for arr in repr_data)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

# 4. Convert + save
tflite_model = converter.convert()
open(TFLITE_OUT, 'wb').write(tflite_model)
```

**INT8 Dequantization (at inference time):**
```python
if output_details[0]["dtype"] == np.int8:
    scale, zero_point = output_details[0]["quantization"]
    out_float = scale * (out_int8.astype(np.float32) - zero_point)
```

**Size Comparison:**

| Stage | File | Size |
|-------|------|------|
| Full Keras model | `cnn_eng.keras` | ~1.6 MB |
| INT8 TFLite | `model_nep.tflite` | ~155–160 KB |
| Compression ratio | — | ~10× |

---

## Phase 5: Feedback ETL

The feedback loop captures user preferences and can inform retraining.

### Trigger

```bash
python csv_extractor.py
```

### Pipeline

```
MySQL: autosuggest_db.feedback
  │   SELECT input, label, rank_score, region, lang FROM feedback
  ▼
ETL/utils/extractor.py   →   pd.DataFrame
  │   pd.read_sql(query, mysql_connection)
  ▼
ETL/utils/transformer.py
  │   df.drop_duplicates()
  │   df.fillna("")
  ▼
ETL/utils/loader.py
  │   df.groupby(["region", "lang"])
  │   → ETL/csv_exports/{region}_{lang}_output.csv
  ▼
CSV files ready for analysis or merging into training data
```

**Export file naming:** `{region}_{lang}_output.csv`  
Examples: `nep_eng_output.csv`, `nep_nep_output.csv`, `ind_hin_output.csv`

**Export format:**
```csv
input,label,rank_score,region,lang
na,नमस्ते,5,nepal,eng
na,नाम,3,nepal,eng
ra,राम्रो,2,nepal,eng
```

### Retraining Cycle

```
feedback CSV → review/curate → merge into train_{lang}.csv
                                        │
                                Phase 1: Preprocessing (update char_map)
                                Phase 2: Trie + Labels (update trie + labels)
                                Phase 3: Retrain CNN model
                                Phase 4: Convert to TFLite
                                        │
                                Restart server (loads new model)
```

---

## Adding a New Language

To add support for a new language (e.g., Tamil for India region):

### 1. Prepare Training Data

```
data_preparation/data/train/tam/train_tam.csv
```

Columns: `input`, `target`

### 2. Add Representative Samples

```
data_preparation/data/samples/tam/repr_samples_tam.txt
```

One word per line for INT8 calibration.

### 3. Update `config.py`

```python
SUPPORTED = {
    ...
    "india": ["hin", "ben", "tam", "tel"],  # already listed
}

FOLDER_KEY = {
    "india": None,   # None = each language uses its own folder
}

LANG_NAMES = {
    "tam": "Tamil",  # already listed
}
```

### 4. Run the Pipeline

```bash
# Preprocessing
python data_preparation/preprocess/preprocesssing.py \
  --train_csv data_preparation/data/train/tam/train_tam.csv

# Trie + Labels (update script path references if needed)
python data_preparation/preprocess/trie.py

# Model Training
python model_trainings/scripts/model_training.py

# TFLite Conversion
python model_trainings/scripts/convert_tflite.py
```

### 5. Start Server with New Language

```bash
set REGION=india
set LANG=tam
python web\backend\autocomplete_api.py
```

---

## File Reference

| File | Phase | Description |
|------|-------|-------------|
| `data/train/{folder}/train_{lang}.csv` | Input | Training data: input+target pairs |
| `data/samples/{folder}/repr_samples_{lang}.txt` | Phase 4 | INT8 calibration samples |
| `data/csv/*.csv` | Source | Raw upstream data |
| `artifacts/{folder}/char_map_{lang}.json` | Phase 1 | Char→int mapping |
| `artifacts/{folder}/trie_{lang}.json` | Phase 2 | Serialized prefix trie |
| `artifacts/{folder}/labels_{lang}.txt` | Phase 2 | Vocabulary list (order = class index) |
| `models/{folder}/cnn_{lang}.keras` | Phase 3 | Full Keras model |
| `models/{folder}/meta_{lang}.json` | Phase 3 | `{"max_len": N}` |
| `models/{folder}/model_{folder}.tflite` | Phase 4 | INT8 TFLite (used at inference) |
| `ETL/csv_exports/{region}_{lang}_output.csv` | Phase 5 | Exported feedback data |
