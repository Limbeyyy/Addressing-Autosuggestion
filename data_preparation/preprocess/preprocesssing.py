import os
import pandas as pd
import argparse
import json
from config import TRAIN_CSV

def build_char_map(samples):
    """Build character-to-index mapping from a list of strings."""
    chars = set()
    for s in samples:
        for c in s:
            chars.add(c)
    chars = sorted(chars)
    char2i = {c: i + 1 for i, c in enumerate(chars)}  # index from 1
    return char2i


def main(args):
    # Resolve artifacts dir relative to this script's location (data_preparation/artifacts)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, "data_preparation", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load dataset (must have input, target columns)
    df = pd.read_csv(args.train_csv, encoding="utf-8")

    # Validate columns
    if "input" not in df.columns or "target" not in df.columns:
        raise ValueError("train.csv must contain 'input' and 'target' columns")

    print("Loaded rows:", len(df))

    # Combine all text for character mapping
    all_samples = list(df["input"].astype(str)) + list(df["target"].astype(str))

    # Build char map
    char_map = build_char_map(all_samples)

    # Save char_map.json
    output_path = os.path.join(artifacts_dir, "char_map.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(char_map, f, ensure_ascii=False, indent=2)

    print(f"Character map saved to {output_path}")
    print(f"Total unique characters: {len(char_map)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        default= TRAIN_CSV,
        help="CSV file with input and target columns"
    )
    args = parser.parse_args()
    main(args)
