# 📡 API Reference — AutoSuggestion Engine

**Base URL (default):** `http://0.0.0.0:8001`  
**Protocol:** HTTP/1.1 (use HTTPS via reverse proxy in production)  
**Content-Type:** `application/json` for all requests and responses

---

## Authentication

All endpoints except `/autocomplete/token` require a **Bearer token** in the `Authorization` header.

### Token Flow

```
Step 1: Request a token (no auth required)
POST /autocomplete/token

Step 2: Use the token for all subsequent requests
Authorization: Bearer <token_from_step_1>
```

### Token Properties

| Property | Value |
|----------|-------|
| Signing algorithm | HS512 (configurable via `ALGORITHM_NAME` in config.py) |
| Encryption | Fernet (AES-128-CBC + HMAC-SHA256) |
| Expiry | 120 minutes (configurable via `TOKEN_ACCESS_TIME`) |
| Bearer format | Opaque Fernet ciphertext (not a raw JWT) |

### Token Error Responses

| Status | Detail | Cause |
|--------|--------|-------|
| `401 Unauthorized` | `"Token expired"` | JWT past its `exp` claim |
| `401 Unauthorized` | `"Invalid token"` | Malformed, tampered, or wrong key |
| `403 Forbidden` | *(FastAPI default)* | Missing `Authorization` header |

---

## Endpoints

---

### `POST /autocomplete/token`

Issue a new JWT access token (Fernet-encrypted). No authentication required.

**Request**
```http
POST /autocomplete/token
Content-Type: application/json
```

No request body required.

**Response `200 OK`**
```json
{
  "access_token": "gAAAAABl...encrypted_fernet_blob...=="
}
```

| Field | Type | Description |
|-------|------|-------------|
| `access_token` | `string` | Fernet-encrypted JWT; pass as Bearer token |

**Example (curl)**
```bash
curl -X POST http://localhost:8001/autocomplete/token
```

**Example (JavaScript)**
```javascript
const res = await fetch("http://localhost:8001/autocomplete/token", {
  method: "POST"
});
const { access_token } = await res.json();
// Store token in memory for subsequent requests
```

---

### `POST /autocomplete/suggest`

Return ranked autocompletion suggestions for a given prefix.

**Request**
```http
POST /autocomplete/suggest
Content-Type: application/json
Authorization: Bearer <access_token>
```

**Request Body**
```json
{
  "text": "na"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ Yes | The typed prefix. Lowercased and stripped server-side. |

**Response `200 OK`**
```json
{
  "data": [
    { "label": "नमस्ते" },
    { "label": "नाम" },
    { "label": "नाइट" }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `data` | `array` | List of suggestion objects, ranked best-first |
| `data[].label` | `string` | Suggested word in target script (e.g., Nepali Unicode) |

**Empty prefix response:**
```json
{ "data": [] }
```

**No candidates response** (prefix not in Trie):
```json
{ "data": [] }
```

**Maximum suggestions returned:** 20 (top_k parameter hardcoded in route handler; up to 500 candidates retrieved from Trie, then ranked by CNN).

**Example (curl)**
```bash
TOKEN="gAAAAABl..."
curl -X POST http://localhost:8001/autocomplete/suggest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text": "nam"}'
```

**Example (JavaScript)**
```javascript
const res = await fetch("http://localhost:8001/autocomplete/suggest", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${access_token}`
  },
  body: JSON.stringify({ text: query })
});
const { data } = await res.json();
// data → [{ label: "नमस्ते" }, ...]
```

**Ranking Logic:**

1. Trie BFS retrieves up to 500 prefix-matching candidates
2. CNN (TFLite) scores each candidate → sorted descending by probability
3. Input-to-target mapping applied (e.g., "namaste" → "नमस्ते") with deduplication
4. MySQL feedback table queried: `SELECT label, rank_score WHERE input LIKE 'prefix%'`
5. Merge: DB results (lower rank_score = higher priority) override model results (fixed score 999)
6. Return top 20

---

### `POST /autocomplete/feedback`

Record that a user selected a specific suggestion. Increments that label's rank score in the database, boosting its future ranking.

**Request**
```http
POST /autocomplete/feedback
Content-Type: application/json
Authorization: Bearer <access_token>
```

**Request Body**
```json
{
  "input": "na",
  "label": "नमस्ते"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | `string` | ✅ Yes | The prefix the user had typed when they made the selection |
| `label` | `string` | ✅ Yes | The suggestion the user clicked or pressed Enter on |

**Response `200 OK`**
```json
{
  "status": "success"
}
```

**Database effect:**
```sql
INSERT INTO feedback (input, label, rank_score)
VALUES ("na", "नमस्ते", 1)
ON DUPLICATE KEY UPDATE rank_score = rank_score + 1;
```

**Example (curl)**
```bash
curl -X POST http://localhost:8001/autocomplete/feedback \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"input": "na", "label": "नमस्ते"}'
```

---

## Error Reference

| HTTP Status | Condition |
|-------------|-----------|
| `200 OK` | All successful responses (including empty suggestion lists) |
| `401 Unauthorized` | Missing, invalid, or expired Bearer token |
| `422 Unprocessable Entity` | Request body missing required fields / wrong types (Pydantic validation) |
| `500 Internal Server Error` | Unhandled server error (check server logs) |

### 422 Example

Request missing the `text` field:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "text"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

---

## FastAPI Interactive Docs

When the server is running locally, FastAPI automatically serves:

| URL | Description |
|-----|-------------|
| `http://localhost:8001/docs` | Swagger UI — interactive API tester |
| `http://localhost:8001/redoc` | ReDoc — readable API docs |
| `http://localhost:8001/openapi.json` | Raw OpenAPI 3.x JSON schema |

---

## Pydantic Schemas

```python
class QueryRequest(BaseModel):
    text: str           # The prefix to autocomplete

class FeedbackRequest(BaseModel):
    input: str          # The prefix that was typed
    label: str          # The label the user selected
```

Response types are plain dicts (not typed Pydantic models); Pydantic validates **input** only.

---

## CORS Configuration

| Mode | Behavior |
|------|----------|
| Development (default) | `CORS_ORIGINS=*` → all origins allowed |
| Production | Set `CORS_ORIGINS=https://yourapp.com` → locked to that origin |

```python
# core/middleware.py
_raw = os.environ.get("CORS_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()] or ["*"]
```

CORS allows all methods and headers (`allow_methods=["*"]`, `allow_headers=["*"]`).
