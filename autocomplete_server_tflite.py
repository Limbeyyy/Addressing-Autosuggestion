# autocomplete_server.py
import json, numpy as np, tensorflow as tf
from collections import deque
from rapidfuzz import process, fuzz
TRIE_PATH = 'python/artifacts/trie.json'
CHAR_MAP = 'python/artifacts/char_map_en.json'
LABELS = 'python/labels_en.txt'
TFLITE_MODEL = 'models/model_en.tflite'
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
                if c == '_end': continue
                dq.append((sub, cur + c))
        return res
trie = json.load(open(TRIE_PATH,encoding='utf-8'))
char_map = json.load(open(CHAR_MAP, encoding='utf-8'))
labels = [l.strip() for l in open(LABELS,encoding='utf-8') if l.strip()]
label2idx = {l:i for i,l in enumerate(labels)}
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
meta = json.load(open('models/meta.json'))
max_len = meta['max_len']
searcher = TrieSearcher(trie)
def encode(s):
    arr = [char_map.get(c,0) for c in s]
    if len(arr) < max_len:
        arr = arr + [0]*(max_len - len(arr))
    else:
        arr = arr[:max_len]
    return np.array(arr, dtype=np.int32)
def score_candidates_tflite(prefix, candidates, top_k=5):
    if not candidates:
        return []
    results = []
    for i, cand in enumerate(candidates):
        arr = np.expand_dims(encode(prefix), axis=0)
        interpreter.set_tensor(input_details[0]['index'], arr.astype(np.int32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            out = scale * (out.astype(np.float32) - zero_point)
        proba = out[0]
        idx = label2idx.get(cand, None)
        score = proba[idx] if idx is not None and idx < len(proba) else 0.0
        results.append((cand, float(score)))
    results.sort(key=lambda x: x[1], reverse=True)
    return [w for w,_ in results[:top_k]]
def suggest(prefix, top_k=5):
    candidates = searcher.autocomplete(prefix, limit=500)
    if not candidates:
        matches = process.extract(prefix, labels, scorer=fuzz.WRatio, limit=top_k)
        return [m[0] for m in matches]
    return score_candidates_tflite(prefix, candidates, top_k)
if __name__ == '__main__':
    while True:
        q = input('prefix (q to quit): ').strip()
        if q.lower() == 'q': break
        print('suggestions:', suggest(q, top_k=5))