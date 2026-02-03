# autocomplete_api.py
import os
import sys
import json
import uvicorn
from core.security import verify_token, create_token
from fastapi import Depends
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import process, fuzz
import pandas as pd
import numpy as np
import tensorflow as tf



# --- 1. SETUP PATHS ---
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Add project root to sys.path
sys.path.append(str(PROJECT_ROOT))

# Database functions
from backend.database.db import update_rank, get_rank, init_db

# ---------------- FILE PATHS ----------------
TRAIN_CSV = PROJECT_ROOT / "prepare_data" / "data" / "train.csv"
CHAR_MAP_PATH = PROJECT_ROOT / "prepare_data" / "python" / "artifacts" / "char_map.json"
TFLITE_MODEL_PATH = PROJECT_ROOT / "model_trainings" / "models" / "model.tflite"
META_JSON_PATH = PROJECT_ROOT / "model_trainings" / "models" / "meta.json"

# ---------------- FASTAPI SETUP ----------------
from fastapi import FastAPI

app = FastAPI(
    title="Autocomplete API",
    version="1.0",
    openapi_url="/autosuggest/openapi.json",  # custom spec URL
    docs_url=None,  # Disable default docs if you want only manual Swagger
)

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATA & MAPPING ----------------
df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
df["input"] = df["input"].astype(str).str.lower().str.strip()
df["target"] = df["target"].str.strip()
eng_to_nep = dict(zip(df["input"], df["target"]))
english_labels = df["input"].tolist()

# ---------------- TRIE CLASS ----------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class TrieSearcher:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_word = True

    def autocomplete(self, prefix, limit=500):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return []
            node = node.children[c]
        res = []
        dq = [(node, prefix)]
        while dq and len(res) < limit:
            nd, cur = dq.pop(0)
            if nd.is_word:
                res.append(cur)
            for ch, child in nd.children.items():
                dq.append((child, cur + ch))
        return res

# ---------------- BUILD TRIE ----------------
searcher = TrieSearcher()
for w in english_labels:
    searcher.insert(w)
print(f"Trie loaded with {len(english_labels)} English words")

# ---------------- LOAD TFLITE MODEL ----------------
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

meta = json.load(open(META_JSON_PATH, encoding='utf-8'))
max_len = meta['max_len']
char_map = json.load(open(CHAR_MAP_PATH, encoding='utf-8'))

# ---------------- ENCODER ----------------
def encode(s):
    arr = [char_map.get(c,0) for c in s]
    if len(arr) < max_len:
        arr += [0]*(max_len - len(arr))
    else:
        arr = arr[:max_len]
    return np.array(arr, dtype=np.int32)

# ---------------- TFLITE SCORING ----------------
def score_candidates_tflite(prefix, candidates, top_k=15):
    if not candidates:
        return []
    results = []
    arr_prefix = np.expand_dims(encode(prefix), axis=0)
    for cand in candidates:
        interpreter.set_tensor(input_details[0]['index'], arr_prefix.astype(np.int32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            out = scale * (out.astype(np.float32) - zero_point)
        proba = out[0]
        idx = english_labels.index(cand) if cand in english_labels else None
        score = proba[idx] if idx is not None and idx < len(proba) else 0.0
        results.append((cand, float(score)))

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate Nepali words while preserving order
    seen = set()
    unique_results = []
    for w,_ in results:
        nep_word = eng_to_nep[w]
        if nep_word not in seen:
            seen.add(nep_word)
            unique_results.append(nep_word)
        if len(unique_results) >= top_k:
            break
    return unique_results

# ---------------- API MODE ----------------
class QueryRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    input: str
    label: str

# ------------------JWT TOKEN ROutes------------------

@app.post("/autocomplete/token")
async def generate_token():
    token = create_token({"user": "yukesh"})
    return {"access_token": token}


@app.post("/autocomplete/suggest")
async def suggestion(q: QueryRequest, user=Depends(verify_token)):
    prefix = q.text.lower().strip()

    # Always return same response structure
    if not prefix:
        return {
            "data": []
        }

    # ---------------- STRICT PREFIX MATCH ----------------
    candidates = searcher.autocomplete(prefix, limit=500)
    if not candidates:
        return {
            "data": []
        }

    # TFLite scoring + English → Nepali
    suggestions = score_candidates_tflite(prefix, candidates, top_k=20)

    if not suggestions:
        return {
            "data": []
        }

    # ---------------- FINAL RESPONSE FORMAT ----------------
    return {
        "data": [
            {"label": item}
            for item in suggestions
        ]
    }


@app.post("/autocomplete/feedback")
async def feedback(f: FeedbackRequest, user=Depends(verify_token)):
    update_rank(f.input, f.label)
    return {"status": "success"}

# ---------------- SERVER START ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
