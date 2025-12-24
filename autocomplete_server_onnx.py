# autocomplete_server_onnx.py
import json
import joblib
import numpy as np
from collections import deque
from rapidfuzz import process, fuzz
import onnxruntime as ort

# -----------------------------
# Paths
# -----------------------------
TRIE_PATH = 'python/artifacts/trie.json'
LABELS = 'python/labels.txt'
VECTORIZER_FILE = 'models/vectorizer.joblib'
ONNX_MODEL = 'models/naive_bayes.onnx'

# -----------------------------
# Trie searcher
# -----------------------------
class TrieSearcher:
    def __init__(self, trie_json):
        self.trie = trie_json

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
            if '_end' in nd:
                res.append(cur)
            for c, sub in nd.items():
                if c == '_end':
                    continue
                dq.append((sub, cur + c))
        return res

# -----------------------------
# Load artifacts
# -----------------------------
trie = json.load(open(TRIE_PATH, encoding='utf-8'))
labels = [l.strip() for l in open(LABELS, encoding='utf-8') if l.strip()]
label2idx = {l: i for i, l in enumerate(labels)}
searcher = TrieSearcher(trie)

# Load vectorizer
vectorizer = joblib.load(VECTORIZER_FILE)

# -----------------------------
# ONNX Runtime setup
# -----------------------------
session = ort.InferenceSession(ONNX_MODEL)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -----------------------------
# Encode input EXACTLY LIKE TRAINING
# -----------------------------
def encode(text):
    """
    Training used DictVectorizer with:
    {"input": <text>}
    So we must use the SAME structure at inference.
    """
    features = {"input": str(text)}
    arr = vectorizer.transform([features]).toarray().astype(np.float32)
    return arr

# -----------------------------
# Scoring candidates using ONNX
# -----------------------------
def score_candidates_onnx(prefix, candidates, top_k=5):
    if not candidates:
        return []

    # ONNX output is predicted class index (int)
    x = encode(prefix)
    pred_idx = session.run([output_name], {input_name: x})[0]

    # If ONNX returns array, convert to scalar
    if isinstance(pred_idx, np.ndarray):
        pred_idx = int(pred_idx[0])
    else:
        pred_idx = int(pred_idx)

    # Get predicted label
    if 0 <= pred_idx < len(labels):
        predicted_label = labels[pred_idx]
    else:
        predicted_label = None

    # Build scoring
    results = []
    for cand in candidates:
        score = 1.0 if cand == predicted_label else 0.0
        results.append((cand, score))

    # If all scores are zero, use fuzzy ranking
    if all(score == 0.0 for _, score in results):
        matches = process.extract(prefix, candidates, scorer=fuzz.WRatio, limit=top_k)
        return [m[0] for m in matches]

    # Otherwise return the best-scored results
    results.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in results[:top_k]]


# -----------------------------
# Suggestion function
# -----------------------------
def suggest(prefix, top_k=5):
    candidates = searcher.autocomplete(prefix, limit=500)
    if not candidates:
        # Fallback fuzzy search
        matches = process.extract(prefix, labels, scorer=fuzz.WRatio, limit=top_k)
        return [m[0] for m in matches]
    return score_candidates_onnx(prefix, candidates, top_k)

# -----------------------------
# Main interactive loop
# -----------------------------
if __name__ == '__main__':
    print("Autocomplete server ready. Type 'q' to quit.")
    while True:
        q = input('prefix: ').strip()
        if q.lower() == 'q':
            break
        suggestions = suggest(q, top_k=5)
        print('Suggestions:', suggestions)
