import os
import logging
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from cryptography.fernet import Fernet
import sys
from pathlib import Path

# ── Ensure project root is on path for config imports ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config import TOKEN_ACCESS_TIME, ALGORITHM_NAME

log = logging.getLogger(__name__)

# ── Secrets from environment (MUST be set before starting the server) ────────
_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
_FERNET_KEY  = os.environ.get("FERNET_KEY")

if not _SECRET_KEY:
    raise ValueError(
        "Environment variable JWT_SECRET_KEY is not set. "
        "Set it before starting the server."
    )
if not _FERNET_KEY:
    raise ValueError(
        "Environment variable FERNET_KEY is not set. "
        "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
    )

ALGORITHM = ALGORITHM_NAME
ACCESS_TOKEN_EXPIRE_MINUTES = TOKEN_ACCESS_TIME

fernet = Fernet(_FERNET_KEY.encode())
security = HTTPBearer()


# ── Fernet helpers ────────────────────────────────────────────────────────────

def encrypt_token(token: str) -> str:
    """Encrypt a JWT string before sending to client."""
    return fernet.encrypt(token.encode()).decode()


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt an encrypted token received from client."""
    return fernet.decrypt(encrypted_token.encode()).decode()


# ── JWT creation ──────────────────────────────────────────────────────────────

def create_token(data: dict) -> str:
    """Create a signed, Fernet-encrypted JWT."""
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload["exp"] = expire
    jwt_token = jwt.encode(payload, _SECRET_KEY, algorithm=ALGORITHM)
    return encrypt_token(jwt_token)


# ── JWT verification ──────────────────────────────────────────────────────────

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Decrypt and verify the JWT; raises 401 on any failure."""
    try:
        token = decrypt_token(credentials.credentials)
        return jwt.decode(token, _SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
