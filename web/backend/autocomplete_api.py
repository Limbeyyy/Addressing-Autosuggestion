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
import datetime
from datetime import datetime, timedelta
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

from config import get_config, PORT, IP_ADDRESS, TOKEN_EXPIRY_SECONDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Config state ──────────────────────────────────────────────────────────────
_CFG: dict | None = None


def _resolve_config(region: str = None, lang: str = None) -> dict:
    """
    Resolve config from:
    1. Explicit region/lang args (from frontend /config call)
    2. REGION + LANG environment variables (production/Docker)
    Raises ValueError if neither is available.
    """
    global _CFG

    r = (region or os.getenv("REGION", "")).strip().lower()
    l = (lang   or os.getenv("LANG",   "")).strip().lower()

    if not r or not l:
        raise ValueError("Region and language are not configured yet. Call /autocomplete/config first.")

    _CFG = get_config(r, l)
    log.info(
        "✔  Config loaded → Region: %s | Language: %s [folder: %s]",
        _CFG["REGION"].capitalize(), _CFG["LANG"].upper(), _CFG.get("FOLDER", _CFG["LANG"]),
    )
    return _CFG


# ── App State ─────────────────────────────────────────────────────────────────
class _State:
    ready:           bool = False
    searcher:        "TrieSearcher | None"       = None
    interpreter:     "tf.lite.Interpreter | None" = None
    input_details:   list = []
    output_details:  list = []
    char_map:        dict = {}
    max_len:         int  = 0
    eng_to_nep:      dict = {}
    english_labels:  list = []


state = _State()


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


# ── Core initializer (called from lifespan OR /config endpoint) ───────────────
async def _initialize_from_config(cfg: dict) -> None:
    """Load all models, trie, and DB using the resolved config dict."""
    train_csv     = cfg["TRAIN_CSV"]
    char_map_path = cfg["CHAR_MAP"]
    tflite_model  = cfg["TFLITE_MODEL"]
    meta_json     = cfg["META_JSON"]

    log.info("Initializing [Region: %s | Language: %s] ...",
             cfg["REGION"].capitalize(), cfg["LANG"].upper())

    # Load CSV
    log.info("Loading dataset: %s", train_csv)
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

    # Char map
    log.info("Loading char map: %s", char_map_path)
    if not os.path.exists(char_map_path):
        raise FileNotFoundError(f"Char map not found: {char_map_path}")
    with open(char_map_path, encoding="utf-8") as f:
        state.char_map = json.load(f)

    # Metadata
    log.info("Loading metadata: %s", meta_json)
    if not os.path.exists(meta_json):
        raise FileNotFoundError(f"Metadata not found: {meta_json}")
    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)
    state.max_len = meta["max_len"]

    # TFLite model
    log.info("Loading TFLite model: %s", tflite_model)
    if not os.path.exists(tflite_model):
        raise FileNotFoundError(f"TFLite model not found: {tflite_model}")
    state.interpreter = tf.lite.Interpreter(model_path=str(tflite_model))
    state.interpreter.allocate_tensors()
    state.input_details  = state.interpreter.get_input_details()
    state.output_details = state.interpreter.get_output_details()
    log.info("TFLite model loaded.")

    # Database
    init_db()

    state.ready = True
    log.info("✔  Initialization complete. Ready to serve requests.")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-init if env vars are already set (production/Docker mode)
    env_region = os.getenv("REGION", "").strip().lower()
    env_lang   = os.getenv("LANG",   "").strip().lower()

    if env_region and env_lang:
        try:
            cfg = _resolve_config(env_region, env_lang)
            await _initialize_from_config(cfg)
        except Exception as e:
            log.error("Auto-init from env vars failed: %s", e)
    else:
        log.info(
            "No REGION/LANG env vars found. "
            "Waiting for frontend to call POST /autocomplete/config ..."
        )

    yield
    log.info("Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

# ── CORS — allow browser requests from file:// and any localhost port ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_middleware(app)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ensure_ready():
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not configured yet. Call POST /autocomplete/config first.",
        )


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


class ConfigRequest(BaseModel):
    region: str
    language: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/autocomplete/status")
async def get_status():
    """
    Returns current configuration status.
    Frontend polls this on load to check if service is ready.
    """
    if state.ready and _CFG:
        return {
            "ready":    True,
            "region":   _CFG.get("REGION", ""),
            "language": _CFG.get("LANG", ""),
        }
    return {"ready": False, "region": None, "language": None}


@app.post("/autocomplete/config")
async def set_config(cfg_req: ConfigRequest):
    """
    Called once from the frontend to configure region + language.
    Triggers full model/trie initialization.
    """
    region = cfg_req.region.strip().lower()
    lang   = cfg_req.language.strip().lower()

    if not region or not lang:
        raise HTTPException(status_code=400, detail="Both 'region' and 'language' are required.")

    # Allow reconfiguration: reset ready flag
    state.ready = False

    try:
        cfg = _resolve_config(region, lang)
        await _initialize_from_config(cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Config initialization failed")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {e}")

    return {
        "status":   "configured",
        "region":   _CFG["REGION"],
        "language": _CFG["LANG"],
    }


@app.get("/autocomplete/token")
async def generate_token():
    """Issue an encrypted JWT access token."""
    _ensure_ready()
    token = create_token({"sub": "api_client"})
    expires_at = datetime.utcnow() + timedelta(seconds=TOKEN_EXPIRY_SECONDS)
    return {
        "access_token": token,
        "expires_at": expires_at.isoformat() + "Z",
        "expires_in": TOKEN_EXPIRY_SECONDS,
    }


@app.post("/autocomplete/suggest")
async def suggestion(q: QueryRequest, user: dict = Depends(verify_token)):
    _ensure_ready()
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
    _ensure_ready()
    update_rank(f.input, f.label)
    return {"status": "success"}


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("autocomplete_api:app", host=IP_ADDRESS, port=PORT, reload=False)