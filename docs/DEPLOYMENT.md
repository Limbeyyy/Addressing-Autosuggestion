# 🚀 Deployment Guide — AutoSuggestion Engine

This guide covers all deployment scenarios from local development to production.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Environment Variables](#environment-variables)
4. [Database Setup](#database-setup)
5. [Starting the Server](#starting-the-server)
6. [Production Deployment](#production-deployment)
7. [Docker Deployment](#docker-deployment)
8. [Nginx Reverse Proxy](#nginx-reverse-proxy)
9. [Systemd Service](#systemd-service)
10. [Health Checks & Monitoring](#health-checks--monitoring)
11. [Security Hardening Checklist](#security-hardening-checklist)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Dependency | Minimum Version | Purpose |
|-----------|----------------|---------|
| Python | 3.10+ | Runtime |
| MySQL | 8.0+ | Feedback storage |
| pip | latest | Package management |
| TensorFlow | 2.16+ | TFLite inference |

---

## Local Development Setup

```bash
# 1. Navigate to project root
cd Autosuggestion_Engine

# 2. Create virtual environment
python -m venv suggest

# 3. Activate virtual environment
#    Windows:
suggest\Scripts\activate
#    Linux / macOS:
source suggest/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Copy and configure environment
#    (Edit .env with your actual secrets - see below)
#    .env already exists; just edit the values

# 6. Run database migration (creates feedback table)
python migration.py

# 7. Start the API server
python web\backend\autocomplete_api.py
#    → Prompts for Region and Language
#    → Server runs at http://0.0.0.0:8001

# 8. Open the frontend
#    Open web/frontend/index.html in a browser
```

---

## Environment Variables

### Required Variables

| Variable | Description | How to Generate |
|----------|-------------|-----------------|
| `JWT_SECRET_KEY` | Signs the JWT token; keep secret | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FERNET_KEY` | Encrypts JWT before sending to clients | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed frontend origins. Use `*` only for development. |
| `REGION` | *(prompts)* | Lock region for non-interactive startup (e.g., `nepal`) |
| `LANG` | *(prompts)* | Lock language for non-interactive startup (e.g., `nep` or `eng`) |

### `.env` File (Local Dev Only)

```env
# ── Security ──────────────────────────────────────────────────────────
JWT_SECRET_KEY=<your-64-char-hex-secret>
FERNET_KEY=<your-fernet-base64-key>=

# ── CORS (use * for local dev) ────────────────────────────────────────
CORS_ORIGINS=*
```

> ⚠️ **NEVER commit `.env` to git.** It is already listed in `.gitignore`.  
> ⚠️ **NEVER use `.env` files in production.** Set env vars via systemd, Docker, or your cloud provider.

### Database Config (in `config.py`)

```python
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "<your-mysql-password>",
    "database": "autosuggest_db",
}
```

> For production, move these values to environment variables and update `config.py` to read from `os.environ`.

---

## Database Setup

### Automatic (Recommended)

The server auto-creates the database and table on first startup:

```python
# db.py calls _ensure_database_exists() at module import time
# Then init_db() is called during FastAPI lifespan startup
```

Just run:
```bash
python migration.py
```

### Manual

```sql
-- Run in MySQL shell
CREATE DATABASE IF NOT EXISTS autosuggest_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE autosuggest_db;

CREATE TABLE IF NOT EXISTS feedback (
    input       VARCHAR(255)  NOT NULL,
    label       VARCHAR(255)  NOT NULL,
    rank_score  INT           NOT NULL DEFAULT 0,
    region      VARCHAR(255)  NOT NULL,
    lang        VARCHAR(255)  NOT NULL,
    PRIMARY KEY (input, label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## Starting the Server

### Development (Interactive Region/Language Prompt)

```bash
python web\backend\autocomplete_api.py
```

### Production (Non-Interactive via Env Vars)

```bash
set REGION=nepal
set LANG=nep
set JWT_SECRET_KEY=<key>
set FERNET_KEY=<key>
set CORS_ORIGINS=https://yourdomain.com

uvicorn web.backend.autocomplete_api:app --host 0.0.0.0 --port 8001 --workers 1
```

> **Use `--workers 1`** — the TFLite model is loaded per worker process during lifespan. Multiple workers will each load their own model, which is fine, but can consume significant memory. Start with 1 and scale if needed.

### With Uvicorn Options

```bash
uvicorn web.backend.autocomplete_api:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1 \
  --log-level info \
  --access-log \
  --timeout-keep-alive 5
```

---

## Production Deployment

### Recommended Stack

```
Client Browser
     │
     ▼
[Nginx / Caddy]  ← TLS termination, static file serving
     │
     ▼
[Uvicorn]  ← ASGI server, port 8001
     │
     ▼
[FastAPI App]  ← autocomplete_api.py
     │
     ▼
[MySQL 8.x]  ← feedback table
```

### Production Checklist

- [ ] Set `CORS_ORIGINS` to your actual frontend domain (not `*`)
- [ ] Set `JWT_SECRET_KEY` to a strong 32-byte random hex string
- [ ] Set `FERNET_KEY` to a generated Fernet key
- [ ] Change default MySQL `root` password or create a dedicated DB user
- [ ] Use TLS (HTTPS) — terminate at Nginx or Caddy
- [ ] Set `REGION` and `LANG` env vars (avoid interactive prompt)
- [ ] Run behind a process manager (systemd, Supervisor, Docker)
- [ ] Monitor logs (structured logging already in place)

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Secrets must be injected at runtime — never bake into image
ENV JWT_SECRET_KEY=""
ENV FERNET_KEY=""
ENV CORS_ORIGINS="*"
ENV REGION="nepal"
ENV LANG="nep"

EXPOSE 8001

CMD ["uvicorn", "web.backend.autocomplete_api:app", \
     "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

### Docker Build & Run

```bash
# Build
docker build -t autosuggest-engine:latest .

# Run (development)
docker run -d \
  --name autosuggest \
  -e JWT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -e FERNET_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
  -e CORS_ORIGINS="*" \
  -e REGION="nepal" \
  -e LANG="nep" \
  -p 8001:8001 \
  autosuggest-engine:latest

# View logs
docker logs -f autosuggest

# Stop
docker stop autosuggest
```

### Docker Compose (with MySQL)

```yaml
# docker-compose.yml
version: "3.9"

services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: autosuggest_db
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    depends_on:
      mysql:
        condition: service_healthy
    environment:
      JWT_SECRET_KEY: "${JWT_SECRET_KEY}"
      FERNET_KEY: "${FERNET_KEY}"
      CORS_ORIGINS: "http://localhost:3000"
      REGION: "nepal"
      LANG: "nep"
    ports:
      - "8001:8001"

volumes:
  mysql_data:
```

```bash
# Set secrets in shell first
export JWT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export FERNET_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

docker-compose up -d
```

---

## Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/autosuggest

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass         http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }

    # Serve static frontend files
    location /static/ {
        alias /app/web/frontend/;
        expires 1d;
    }
}
```

```bash
# Enable site and restart
sudo ln -s /etc/nginx/sites-available/autosuggest /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Systemd Service

```ini
# /etc/systemd/system/autosuggest.service

[Unit]
Description=AutoSuggestion Engine API
After=network.target mysql.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/autosuggest
ExecStart=/opt/autosuggest/suggest/bin/uvicorn web.backend.autocomplete_api:app \
          --host 127.0.0.1 --port 8001 --workers 1

# Secrets from environment
EnvironmentFile=/etc/autosuggest/environment

Restart=on-failure
RestartSec=5

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/autosuggest/environment  (chmod 600, owned by root)
JWT_SECRET_KEY=<your-key>
FERNET_KEY=<your-key>
CORS_ORIGINS=https://yourdomain.com
REGION=nepal
LANG=nep
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable autosuggest
sudo systemctl start autosuggest
sudo systemctl status autosuggest
```

---

## Health Checks & Monitoring

### Basic Health Check

```bash
# Check server is responding
curl http://localhost:8001/docs
# → FastAPI Swagger UI (200 OK)
```

### Token Round-Trip Test

```bash
# Get token
TOKEN=$(curl -s -X POST http://localhost:8001/autocomplete/token | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Test suggest
curl -X POST http://localhost:8001/autocomplete/suggest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text": "na"}'
# → {"data": [{"label": "..."}, ...]}
```

### Application Logs

The server uses Python's `logging` module with `INFO` level:

```
2026-02-24 09:00:00 [INFO] Starting up [Region: Nepal | Language: NEP] ...
2026-02-24 09:00:01 [INFO] Dataset loaded: 5000 rows
2026-02-24 09:00:01 [INFO] Trie built with 5000 words
2026-02-24 09:00:02 [INFO] TFLite model loaded.
2026-02-24 09:00:02 [INFO] Database initialized successfully.
2026-02-24 09:00:02 [INFO] Startup complete. Ready to serve requests.
```

To increase log verbosity:
```bash
uvicorn ... --log-level debug
```

---

## Security Hardening Checklist

| Item | Action |
|------|--------|
| ✅ JWT secret | Minimum 32 bytes, cryptographically random |
| ✅ Fernet key | Generated via `Fernet.generate_key()`, never reused |
| ✅ CORS | Set `CORS_ORIGINS` to exact frontend domain in production |
| ✅ HTTPS | Terminate TLS at Nginx/Caddy; never send tokens over HTTP in production |
| ✅ DB credentials | Use a dedicated MySQL user with only SELECT/INSERT/UPDATE permissions on `autosuggest_db` |
| ✅ `.env` not committed | `.gitignore` already excludes `.env` |
| ✅ `config.py` not committed | `.gitignore` already excludes `config.py` |
| ✅ Token expiry | Default 120 min; reduce for higher security requirements |
| ⚠️ DB password in config.py | Move to env var in production |
| ⚠️ Frontend token handling | Current `app.js` doesn't implement token auth; update for production |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ValueError: Environment variable JWT_SECRET_KEY is not set` | `.env` not loaded or empty | Check `.env` exists; `source .env` or use `python-dotenv` |
| `FileNotFoundError: Training CSV not found` | Wrong region/lang or missing training data | Check `TRAIN_CSV` path via `config.get_config(region, lang)["TRAIN_CSV"]` |
| `FileNotFoundError: TFLite model not found` | Model not trained or wrong folder | Run training pipeline; check `model_trainings/models/{folder}/` |
| `mysql.connector.errors.DatabaseError` | MySQL not running or wrong credentials | Start MySQL; verify `DB_CONFIG` in `config.py` |
| Double interactive prompt | Uvicorn re-importing module | Use `REGION`+`LANG` env vars; or run via `python autocomplete_api.py` (handles this internally) |
| `401 Unauthorized` on all requests | Token expired or wrong key | Re-fetch token via `POST /autocomplete/token` |
| Empty suggestion list always | Prefix not in trie | Check that training CSV is loaded; try different prefix |
| CORS errors in browser | Missing or wrong `CORS_ORIGINS` | Set `CORS_ORIGINS=http://localhost:PORT` or `*` for dev |
