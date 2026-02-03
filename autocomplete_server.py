# autocomplete_server.py
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
from rapidfuzz import process, fuzz

# ---------------- FILE PATHS ----------------
TRIE_PATH = r'C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\prepare_data\python\artifacts\trie.json'  # old, unused
CHAR_MAP = r'C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\prepare_data\python\artifacts\char_map.json'
TFLITE_MODEL = r'C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\model_trainings\models\model.tflite'
TRAIN_CSV = r"C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\prepare_data\data\train.csv"
META_JSON = r"C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\model_trainings\models\meta.json"

# ---------------- LOAD DATASET ----------------
df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
df["input"] = df["input"].astype(str).str.lower().str.strip()
df["target"] = df["target"].str.strip()

# English → Nepali dictionary
eng_to_nep = dict(zip(df["input"], df["target"]))
english_labels = df["input"].tolist()

# ---------------- TRIE CLASS ----------------
class TrieSearcher:
    def __init__(self):
        self.trie = {}

    def insert(self, word):
        node = self.trie
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node["_end"] = True

    def autocomplete(self, prefix, limit=500):
        node = self.trie
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        res = []
        dq = deque([(node, prefix)])
        while dq and len(res) < limit:
            nd, cur = dq.popleft()
            if "_end" in nd:
                res.append(cur)
            for c, sub in nd.items():
                if c == "_end": 
                    continue
                dq.append((sub, cur + c))
        return res

# ---------------- BUILD TRIE ----------------
searcher = TrieSearcher()
for w in english_labels:
    searcher.insert(w)
print(f"Trie loaded with {len(english_labels)} English words")

# ---------------- LOAD TFLITE MODEL ----------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

meta = json.load(open(META_JSON, encoding='utf-8'))
max_len = meta['max_len']

char_map = json.load(open(CHAR_MAP, encoding='utf-8'))

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

    # remove duplicate Nepali suggestions
    seen = set()
    final = []
    for w, _ in results:
        nep = eng_to_nep[w]
        if nep not in seen:
            seen.add(nep)
            final.append(nep)
        if len(final) >= top_k:
            break

    return final

import mysql.connector
from config import DB_CONFIG

#MySQL Database fetch function
def fetch_db_suggestions(prefix, limit=50):
    prefix = prefix.lower().strip()
    if not prefix:
        return []

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
        SELECT label, rank_score
        FROM feedback
        WHERE input LIKE %s
        ORDER BY rank_score ASC
        LIMIT %s
    """

    cur.execute(query, (prefix + "%", int(limit)))
    rows = cur.fetchall()

    print("DB PREFIX:", prefix)
    print("DB ROWS:", rows)


    cur.close()
    conn.close()

    # return list of tuples: [(label, rank), ...]
    return [(label, rank) for label, rank in rows]

# ---------------- SUGGEST FUNCTION ----------------
def suggest(prefix, top_k=50):
    prefix = prefix.lower().strip()
    if not prefix:
        return {"message": "No input provided", "suggestions": []}

    # ---------------- MODEL SUGGESTIONS ----------------
    candidates = searcher.autocomplete(prefix, limit=500)
    model_suggestions = []
    if candidates:
        model_suggestions = score_candidates_tflite(prefix, candidates, top_k)

    # Convert model output to ranked format
    model_ranked = [(s, 999) for s in model_suggestions]  # model rank = low priority

    # ---------------- DATABASE SUGGESTIONS ----------------
    db_suggestions = fetch_db_suggestions(prefix, limit=top_k)
    # db_suggestions → [(label, rank), ...]

    # ---------------- MERGE ----------------
    combined = {}
    
    # DB first (higher priority)
    for label, rank in db_suggestions:
        combined[label] = rank

    # Then model suggestions
    for label, rank in model_ranked:
        if label not in combined:
            combined[label] = rank

    # ---------------- SORT BY RANK ----------------
    final_sorted = sorted(combined.items(), key=lambda x: x[1])

    # Extract only labels
    final_labels = [label for label, _ in final_sorted][:top_k]

    if not final_labels:
        return {"message": f"No suggestions found for '{prefix}'", "suggestions": []}

    return {"suggestions": final_labels}


# ---------------- DEBUG CLI ----------------
if __name__ == '__main__':
    while True:
        q = input('prefix (q to quit): ').strip()
        if q.lower() == 'q': break
        print('suggestions:', suggest(q, top_k=50))