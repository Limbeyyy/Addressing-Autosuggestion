# 📡 API Reference — AutoSuggestion Engine

**Base URL (default):** `http://0.0.0.0:8001`  
**Protocol:** HTTP/1.1 (use HTTPS via reverse proxy in production)  
**Content-Type:** `application/json` for all requests and responses

---

## Authentication

Endpoints `/autocomplete/suggest` and `/autocomplete/feedback` require a **Bearer token** in the `Authorization` header.  
`/autocomplete/status`, `/autocomplete/config`, and `/autocomplete/token` do **not** require authentication.

### Token Flow

```
Step 1: (Optional) Configure region + language
POST /autocomplete/config  {"region": "nep", "language": "eng"}

Step 2: Request a token (no auth required)
GET /autocomplete/token

Step 3: Use the token for all subsequent requests
Authorization: Bearer <token_from_step_2>
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

### `GET /autocomplete/status`

Returns the current service readiness and active configuration. The frontend polls this on load to determine if the backend is already configured.

**Request**
```http
GET /autocomplete/status
```

No request body or auth required.

**Response `200 OK` — when ready**
```json
{
  "ready": true,
  "region": "nep",
  "language": "eng"
}
```

**Response `200 OK` — when not yet configured**
```json
{
  "ready": false,
  "region": null,
  "language": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `ready` | `boolean` | `true` if model, trie, and DB are fully initialized |
| `region` | `string \| null` | Active region code (e.g., `"nep"`) or `null` |
| `language` | `string \| null` | Active language code (e.g., `"eng"`) or `null` |

**Example (curl)**
```bash
curl http://localhost:8001/autocomplete/status
```

**Example (JavaScript)**
```javascript
const res = await fetch("http://localhost:8001/autocomplete/status");
const { ready, region, language } = await res.json();
if (!ready) {
  // trigger POST /autocomplete/config
}
```

---

### `POST /autocomplete/config`

Configure the service's active region and language. Triggers full initialization: loads the training CSV, builds the Trie, loads the char map and TFLite model, and connects to the database.

Call this once after startup when `REGION`/`LANG` env vars are not set (i.e., when `/autocomplete/status` returns `{"ready": false}`). Can also be called again to **reconfigure** to a different region/language at runtime.

**Request**
```http
POST /autocomplete/config
Content-Type: application/json
```

**Request Body**
```json
{
  "region": "nep",
  "language": "eng"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `region` | `string` | ✅ Yes | Region code — must be one of `"nep"`, `"ind"`, `"global"` |
| `language` | `string` | ✅ Yes | Language code valid for the given region (see table below) |

**Supported region + language combinations:**

| Region | Valid language codes |
|--------|---------------------|
| `nep` | `nep`, `eng` |
| `ind` | `hin`, `kannada`, `tamil`, `sanskrit`, `punjabi`, `gujarati` |
| `global` | `eng` |

**Response `200 OK`**
```json
{
  "status": "configured",
  "region": "nep",
  "language": "eng"
}
```

**Error Responses**

| Status | Detail | Cause |
|--------|--------|-------|
| `400 Bad Request` | `"Both 'region' and 'language' are required."` | Empty/missing fields |
| `400 Bad Request` | `"Unsupported region '...' ..."` | Region not in `SUPPORTED` |
| `400 Bad Request` | `"Language '...' is not available for region '...'"` | Invalid language for region |
| `404 Not Found` | `"Training CSV not found: ..."` | Model/artifact file missing on disk |
| `500 Internal Server Error` | `"Initialization failed: ..."` | Any other startup error |

**Example (curl)**
```bash
curl -X POST http://localhost:8001/autocomplete/config \
  -H "Content-Type: application/json" \
  -d '{"region": "nep", "language": "eng"}'
```

**Example (JavaScript)**
```javascript
const res = await fetch("http://localhost:8001/autocomplete/config", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ region: "nep", language: "eng" })
});
const { status, region, language } = await res.json();
```

---

### `GET /autocomplete/token`

Issue a new JWT access token (Fernet-encrypted). No authentication required. Requires the service to be configured and ready first (call `POST /autocomplete/config` if needed).

**Request**
```http
GET /autocomplete/token
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

**Error Responses**

| Status | Detail | Cause |
|--------|--------|-------|
| `503 Service Unavailable` | `"Service not configured yet. Call POST /autocomplete/config first."` | Service not initialized |

**Example (curl)**
```bash
curl http://localhost:8001/autocomplete/token
```

**Example (JavaScript)**
```javascript
const res = await fetch("http://localhost:8001/autocomplete/token");
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

**Maximum suggestions returned:** 20 (top_k parameter; up to 500 candidates retrieved from Trie, then ranked by CNN).

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
3. Input-to-target mapping applied (e.g., `"namaste"` → `"नमस्ते"`) with deduplication
4. DB feedback merged: rows with lower `rank_score` take priority over model score (999)
5. Return top 20

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
| `400 Bad Request` | Invalid region/language in `/config`, or missing required fields |
| `401 Unauthorized` | Missing, invalid, or expired Bearer token |
| `404 Not Found` | Required model/artifact file not found on disk |
| `422 Unprocessable Entity` | Request body missing required fields / wrong types (Pydantic validation) |
| `503 Service Unavailable` | Service not initialized — call `POST /autocomplete/config` first |
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

class ConfigRequest(BaseModel):
    region: str         # Region code: "nep", "ind", "global"
    language: str       # Language code valid for the region
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
CORS credentials are enabled (`allow_credentials=True`).
