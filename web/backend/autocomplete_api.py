# Environment Call
from dotenv import load_dotenv
load_dotenv()

# Imports
import os
import sys
import json
import logging
import uvicorn
from contextlib import asynccontextmanager
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.security import verify_token, create_token
from core.middleware import setup_middleware
from database.db import update_rank, get_rank, init_db

# ── Ensure project root is on path for config imports ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import get_config, prompt_region_and_language, PORT, IP_ADDRESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Deferred config resolver ──────────────────────────────────────────────────
#
#  Config is resolved ONCE inside lifespan (not at import time) to avoid the
#  double-prompt caused by uvicorn re-importing the module when launched via
#  `python autocomplete_api.py` (parent process imports → spawns worker which
#  re-imports the same module at module level).
#
#  Two modes (checked in order):
#  1. ENV VARS  – REGION + LANG set before launch → no prompt, ideal for
#                 production / Docker / systemd.
#  2. INTERACTIVE – env vars absent → user is prompted once in the terminal.
#
#  Examples:
#    REGION=nepal LANG=eng uvicorn autocomplete_api:app --host 0.0.0.0 ...
#    REGION=nepal LANG=nep python autocomplete_api.py
# ─────────────────────────────────────────────────────────────────────────────
_CFG: dict | None = None   # populated once by _resolve_config()


def _resolve_config() -> dict:
    """Return the config dict, resolving it only on the first call."""
    global _CFG
    if _CFG is not None:
        return _CFG

    env_region = os.getenv("REGION", "").strip().lower()
    env_lang   = os.getenv("LANG",   "").strip().lower()

    if env_region and env_lang:
        log.info("Config from env vars → REGION=%s  LANG=%s", env_region, env_lang)
        _CFG = get_config(env_region, env_lang)
        log.info(
            "✔  Config loaded → Region: %s | Language: %s [folder: %s]",
            _CFG["REGION"].capitalize(), _CFG["LANG"].upper(), _CFG.get("FOLDER", _CFG["LANG"]),
        )
    else:
        _CFG = prompt_region_and_language()  

    return _CFG


# ── Trie ──────────────────────────────────────────────────────────────────────
class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children: dict = {}
        self.is_word: bool = False


class TrieSearcher:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True

    def autocomplete(self, prefix: str, limit: int = 500) -> list[str]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: list[str] = []
        queue = deque([(node, prefix)])
        while queue and len(results) < limit:
            nd, cur = queue.popleft()
            if nd.is_word:
                results.append(cur)
            for ch, child in nd.children.items():
                queue.append((child, cur + ch))
        return results


# ── App state (populated during lifespan) ────────────────────────────────────
class _State:
    searcher: TrieSearcher
    interpreter: tf.lite.Interpreter
    input_details: list
    output_details: list
    char_map: dict
    max_len: int
    eng_to_nep: dict
    english_labels: list[str]


state = _State()


# ── Lifespan: all startup/shutdown I/O ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Resolve config here (first and only time) ─────────────────────────────
    cfg          = _resolve_config()
    train_csv    = cfg["TRAIN_CSV"]
    char_map_path= cfg["CHAR_MAP"]
    tflite_model = cfg["TFLITE_MODEL"]
    meta_json    = cfg["META_JSON"]

    log.info(
        "Starting up [Region: %s | Language: %s] ...",
        cfg["REGION"].capitalize(),
        cfg["LANG"].upper(),
    )

    # Load CSV data
    log.info("Loading dataset from: %s", train_csv)
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    df = pd.read_csv(train_csv, encoding="utf-8")
    df["input"]  = df["input"].astype(str).str.lower().str.strip()
    df["target"] = df["target"].str.strip()
    state.eng_to_nep     = dict(zip(df["input"], df["target"]))
    state.english_labels = df["input"].tolist()
    log.info("Dataset loaded: %d rows", len(df))

    # Build trie
    state.searcher = TrieSearcher()
    for word in state.english_labels:
        state.searcher.insert(word)
    log.info("Trie built with %d words", len(state.english_labels))

    # Load char map
    log.info("Loading char map from: %s", char_map_path)
    if not os.path.exists(char_map_path):
        raise FileNotFoundError(f"Char map not found: {char_map_path}")
    with open(char_map_path, encoding="utf-8") as f:
        state.char_map = json.load(f)

    # Load metadata
    log.info("Loading metadata from: %s", meta_json)
    if not os.path.exists(meta_json):
        raise FileNotFoundError(f"Metadata not found: {meta_json}")
    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)
    state.max_len = meta["max_len"]

    # Load TFLite model
    log.info("Loading TFLite model from: %s", tflite_model)
    if not os.path.exists(tflite_model):
        raise FileNotFoundError(f"TFLite model not found: {tflite_model}")
    state.interpreter = tf.lite.Interpreter(model_path=str(tflite_model))
    state.interpreter.allocate_tensors()
    state.input_details  = state.interpreter.get_input_details()
    state.output_details = state.interpreter.get_output_details()
    log.info("TFLite model loaded.")

    # Initialize database
    init_db()

    log.info("Startup complete. Ready to serve requests.")
    yield
    log.info("Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

setup_middleware(app) 

# ── Helpers ───────────────────────────────────────────────────────────────────
def _encode(s: str) -> np.ndarray:
    arr = [state.char_map.get(c, 0) for c in s]
    arr = arr[:state.max_len] if len(arr) >= state.max_len else arr + [0] * (state.max_len - len(arr))
    return np.array(arr, dtype=np.int32)


def _score_candidates(prefix: str, candidates: list[str], top_k: int = 20) -> list[str]:
    if not candidates:
        return []
    arr_prefix = np.expand_dims(_encode(prefix), axis=0)
    results = []
    for cand in candidates:
        state.interpreter.set_tensor(state.input_details[0]["index"], arr_prefix.astype(np.int32))
        state.interpreter.invoke()
        out = state.interpreter.get_tensor(state.output_details[0]["index"])
        if state.output_details[0]["dtype"] == np.int8:
            scale, zero_point = state.output_details[0]["quantization"]
            out = scale * (out.astype(np.float32) - zero_point)
        proba = out[0]
        idx   = state.english_labels.index(cand) if cand in state.english_labels else None
        score = float(proba[idx]) if idx is not None and idx < len(proba) else 0.0
        results.append((cand, score))

    results.sort(key=lambda x: x[1], reverse=True)

    seen:   set       = set()
    unique: list[str] = []
    for word, _ in results:
        nep = state.eng_to_nep[word]
        if nep not in seen:
            seen.add(nep)
            unique.append(nep)
        if len(unique) >= top_k:
            break
    return unique


# ── Request/Response schemas ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    input: str
    label: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/autocomplete/token")
async def generate_token():
    """Issue an encrypted JWT access token."""
    token = create_token({"sub": "api_client"})
    return {"access_token": token}


@app.post("/autocomplete/suggest")
async def suggestion(q: QueryRequest, user: dict = Depends(verify_token)):
    prefix = q.text.lower().strip()
    if not prefix:
        return {"data": []}

    candidates = state.searcher.autocomplete(prefix, limit=500)
    if not candidates:
        return {"data": []}

    suggestions = _score_candidates(prefix, candidates, top_k=20)
    return {"data": [{"label": item} for item in suggestions]}


@app.post("/autocomplete/feedback")
async def feedback(f: FeedbackRequest, user: dict = Depends(verify_token)):
    update_rank(f.input, f.label)
    return {"status": "success"}


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve config here (in the parent/launcher process) so that when uvicorn
    # imports this module again internally the env vars are already set and
    # _resolve_config() skips the prompt entirely → prompt appears exactly once.
    cfg = _resolve_config()
    os.environ["REGION"] = cfg["REGION"]   # e.g. "nepal"
    os.environ["LANG"]   = cfg["LANG"]     # e.g. "nep" or "eng"

    