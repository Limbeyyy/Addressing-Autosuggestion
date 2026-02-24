# transformer.py
import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional data transformations:
    - Remove duplicates
    - Fill missing values
    """
    df = df.drop_duplicates()
    df = df.fillna("")  # Replace NaN with empty string
    return df