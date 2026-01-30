import mysql.connector
from config import DB_CONFIG


# --- Get a database connection ---
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

# --- Initialize database and table ---
def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            input VARCHAR(255),
            label VARCHAR(255),
            rank_score INT DEFAULT 0,
            PRIMARY KEY(input, label)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully.")

# --- Update rank_score for a given input-label pair ---
def update_rank(inp, label):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (input, label, rank_score)
        VALUES (%s, %s, 1)
        ON DUPLICATE KEY UPDATE rank_score = rank_score + 1
    """, (inp, label))
    conn.commit()
    cursor.close()
    conn.close()

# --- Get total rank_score for a label ---
def get_rank(label):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(rank_score) FROM feedback WHERE label=%s", (label,))
    result = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return result or 0
