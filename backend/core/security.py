# ---------------- API SECURITY (JWT + ENCRYPTION) ----------------

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from core.config import GENERATED_FERNET_KEY     


# ---------------- CONFIG ----------------

SECRET_KEY = "9741683575@@!!!"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# ⚠️ Generate once and store securely (env variable in production)
FERNET_KEY = GENERATED_FERNET_KEY
fernet = Fernet(FERNET_KEY)

security = HTTPBearer()

# ---------------- TOKEN HELPERS ----------------

def encrypt_token(token: str) -> str:
    """Encrypt JWT before sending to client"""
    return fernet.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Decrypt token received from client"""
    return fernet.decrypt(encrypted_token.encode()).decode()

# ---------------- JWT CREATION ----------------

def create_token(data: dict) -> str:
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    jwt_token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    # 🔐 Encrypt JWT
    return encrypt_token(jwt_token)

# ---------------- JWT VERIFICATION ----------------

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        encrypted_token = credentials.credentials

        # 🔓 Decrypt first
        token = decrypt_token(encrypted_token)

        # ✅ Verify JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
