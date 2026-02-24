import os
import logging
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)

# Allow locking down origins in production via env var:
# CORS_ORIGINS="https://yourdesiredlinks"
# Leave unset to allow all origins (useful during development).
_raw = os.environ.get("CORS_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()] or ["*"]


def setup_middleware(app):
    if ALLOWED_ORIGINS == ["*"]:
        log.warning("CORS is open to all origins. Set CORS_ORIGINS env var in production.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
