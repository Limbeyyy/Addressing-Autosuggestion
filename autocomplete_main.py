import json
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
import mysql.connector

from config import prompt_region_and_language

# ─────────────────────────────────────────────────────────────
#  STEP 1 – Ask user for region & language, resolve all paths
# ─────────────────────────────────────────────────────────────
CFG = prompt_region_and_language()

TRIE_PATH    = CFG["TRIE_PATH"]
CHAR_MAP     = CFG["CHAR_MAP"]
TFLITE_MODEL = CFG["TFLITE_MODEL"]
TRAIN_CSV    = CFG["TRAIN_CSV"]
META_JSON    = CFG["META_JSON"]
DB_CONFIG    = CFG["DB_CONFIG"]

# ─────────────────────────────────────────────────────────────
#  STEP 2 – Load dataset
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
df["input"]  = df["input"].astype(str).str.lower().str.strip()
df["target"] = df["target"].str.strip()

eng_to_nep    = dict(zip(df["input"], df["target"]))
english_labels = df["input"].tolist()

# ─────────────────────────────────────────────────────────────
#  TRIE
# ─────────────────────────────────────────────────────────────
class TrieSearcher:
    def __init__(self):
        self.trie = {}

    def insert(self, word: str):
        node = self.trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["_end"] = True

    def autocomplete(self, prefix: str, limit: int = 500):
        node = self.trie
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        results = []
        dq = deque([(node, prefix)])
        while dq and len(results) < limit:
            nd, cur = dq.popleft()
            if "_end" in nd:
                results.append(cur)
            for c, sub in nd.items():
                if c != "_end":
                    dq.append((sub, cur + c))
        return results


searcher = TrieSearcher()
for w in english_labels:
    searcher.insert(w)
print(f"[Trie] Loaded {len(english_labels)} words  "
      f"({CFG['LANG'].upper()} / {CFG['REGION'].capitalize()})")

# ─────────────────────────────────────────────────────────────
#  TFLite model
# ─────────────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

meta    = json.load(open(META_JSON, encoding="utf-8"))
max_len = meta["max_len"]

char_map = json.load(open(CHAR_MAP, encoding="utf-8"))

# ─────────────────────────────────────────────────────────────
#  Encoder
# ─────────────────────────────────────────────────────────────
def encode(s: str) -> np.ndarray:
    arr = [char_map.get(c, 0) for c in s]
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    else:
        arr = arr[:max_len]
    return np.array(arr, dtype=np.int32)

# ─────────────────────────────────────────────────────────────
#  TFLite scoring
# ─────────────────────────────────────────────────────────────
def score_candidates_tflite(prefix: str, candidates: list, top_k: int = 15) -> list:
    if not candidates:
        return []

    arr_prefix = np.expand_dims(encode(prefix), axis=0).astype(np.int32)
    results    = []

    for cand in candidates:
        interpreter.set_tensor(input_details[0]["index"], arr_prefix)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            out = scale * (out.astype(np.float32) - zero_point)

        proba = out[0]
        idx   = english_labels.index(cand) if cand in english_labels else None
        score = float(proba[idx]) if (idx is not None and idx < len(proba)) else 0.0
        results.append((cand, score))

    results.sort(key=lambda x: x[1], reverse=True)

    seen, final = set(), []
    for w, _ in results:
        nep = eng_to_nep[w]
        if nep not in seen:
            seen.add(nep)
            final.append(nep)
        if len(final) >= top_k:
            break

    return final

# ─────────────────────────────────────────────────────────────
#  Database fetch
# ─────────────────────────────────────────────────────────────
def fetch_db_suggestions(prefix: str, limit: int = 50) -> list:
    prefix = prefix.lower().strip()
    if not prefix:
        return []

    conn = mysql.connector.connect(**DB_CONFIG)
    cur  = conn.cursor()

    query = """
        SELECT label, rank_score
        FROM feedback
        WHERE input LIKE %s
        ORDER BY rank_score ASC
        LIMIT %s
    """
    cur.execute(query, (prefix + "%", int(limit)))
    rows = cur.fetchall()

    print(f"[DB] prefix={prefix!r}  rows={rows}")

    cur.close()
    conn.close()
    return [(label, rank) for label, rank in rows]

# ─────────────────────────────────────────────────────────────
#  Main suggest function
# ─────────────────────────────────────────────────────────────
def suggest(prefix: str, top_k: int = 50) -> dict:
    prefix = prefix.lower().strip()
    if not prefix:
        return {"message": "No input provided", "suggestions": []}

    # -- Model suggestions --
    candidates        = searcher.autocomplete(prefix, limit=500)
    model_suggestions = score_candidates_tflite(prefix, candidates, top_k) if candidates else []
    model_ranked      = [(s, 999) for s in model_suggestions]

    # -- Database suggestions --
    db_suggestions = fetch_db_suggestions(prefix, limit=top_k)

    # -- Merge (DB has higher priority / lower rank score) --
    combined = {}
    for label, rank in db_suggestions:
        combined[label] = rank
    for label, rank in model_ranked:
        if label not in combined:
            combined[label] = rank

    final_labels = [lbl for lbl, _ in sorted(combined.items(), key=lambda x: x[1])][:top_k]

    if not final_labels:
        return {"message": f"No suggestions found for '{prefix}'", "suggestions": []}

    return {"suggestions": final_labels}

# ─────────────────────────────────────────────────────────────
#  Debug CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nEngine ready  [{CFG['LANG'].upper()} | {CFG['REGION'].capitalize()}]")
    print("Type a prefix to get suggestions, or 'q' to quit.\n")
    while True:
        q = input("prefix > ").strip()
        if q.lower() == "q":
            break
        result = suggest(q, top_k=100)
        if "message" in result:
            print(" ", result["message"])
        else:
            print("  Suggestions:", result["suggestions"])