import sys
from pathlib import Path
import mysql.connector
import pandas as pd

# ── Ensure project root is on path for config imports ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import DB_CONFIG

def extract_data(query: str) -> pd.DataFrame:
    """
    Connects to the database, executes the query, and returns a pandas DataFrame.
    """
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"]
        )
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return pd.DataFrame()  