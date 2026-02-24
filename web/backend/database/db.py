import logging
import sys
import os
import mysql.connector.pooling

# Ensure the project root (3 levels up from this file) is on sys.path
# so that `config.py` at the root is always importable, regardless of
# whether this module is run directly or imported as part of a package.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DB_CONFIG

log = logging.getLogger(__name__)


def _ensure_database_exists() -> None:
    """Create the database if it does not already exist."""
    db_name = DB_CONFIG["database"]
    # Build a config WITHOUT the database key so we can connect to MySQL itself
    server_cfg = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    tmp_conn = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="_init_pool", pool_size=1, **server_cfg
    ).get_connection()
    try:
        cursor = tmp_conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        cursor.close()
        log.info("Database '%s' ensured.", db_name)
    finally:
        tmp_conn.close()


_ensure_database_exists()

# ── Connection pool (created once at module load) ─────────────────────────────
_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="autocomplete_pool",
    pool_size=5,
    **DB_CONFIG,
)


def get_connection():
    """Get a connection from the pool."""
    return _pool.get_connection()


def init_db():
    """Create the feedback table if it does not exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                input      VARCHAR(255)  NOT NULL,
                label      VARCHAR(255)  NOT NULL,
                rank_score INT           NOT NULL DEFAULT 0,
                region     VARCHAR(255)  NOT NULL,
                lang     VARCHAR(255)  NOT NULL,
                PRIMARY KEY (input, label)
            )
        """)
        conn.commit()
        cursor.close()
        log.info("Database initialized successfully.")
    finally:
        conn.close()


def update_rank(inp: str, label: str) -> None:
    """Increment rank_score for an input-label pair (upsert)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (input, label, rank_score)
            VALUES (%s, %s, 1)
            ON DUPLICATE KEY UPDATE rank_score = rank_score + 1
        """, (inp, label))
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def get_rank(label: str) -> int:
    """Return the total rank_score across all inputs for a given label."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT SUM(rank_score) FROM feedback WHERE label = %s", (label,)
        )
        result = cursor.fetchone()[0]
        cursor.close()
        return int(result) if result else 0
    finally:
        conn.close()
