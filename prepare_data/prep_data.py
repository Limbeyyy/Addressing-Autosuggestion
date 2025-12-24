# prepare_data/prepare_data.py
import os
import json
import argparse
import pandas as pd

# from prepare_data.utils import parse_list
from prepare_data.generator_rules import apply_rule_by_name, double_letter, swap_adjacent
from prepare_data.phonetic_rules import phonetic_variants
from prepare_data.keyboard_rules import keyboard_neighbors
from prepare_data.neural_synthetic import generate_neural_like
from prepare_data.morpheme_generator import generate_prefix_variants
from prepare_data.utils import parse_list

# ensure reproducible
import random
random.seed(42)

def expand_row(row):
    target = str(row.get("label", "")).strip()
    if not target or target.lower() == "nan":
        return []

    pairs = set()

    # 0️⃣ Always include full word as feature
    pairs.add((target, target))

    # 1️⃣ Prefix features
    for i in range(1, 5):
        key = f"prefix_{i}"
        val = row.get(key)

        if pd.notna(val):
            val = str(val).strip()
            if val:
                pairs.add((val, target))

    # 2️⃣ Typo features
    raw_typos = row.get("typos")

    if isinstance(raw_typos, str) and raw_typos.strip():
        # JSON-style list
        if raw_typos.strip().startswith("["):
            typos = parse_list(raw_typos)
        else:
            # comma-separated fallback
            typos = [t.strip() for t in raw_typos.split(",") if t.strip()]

        for t in typos:
            pairs.add((t, target))

    return list(pairs)


    
def main(args):
    os.makedirs('data', exist_ok=True)
    os.makedirs('python/artifacts', exist_ok=True)

    df = pd.read_csv(args.prefix_csv, encoding='utf-8')

    all_pairs = []
    for _, row in df.iterrows():
        pairs = expand_row(row)
        all_pairs.extend(pairs)

    # create dataframe
    out_df = pd.DataFrame(all_pairs, columns=["input", "target"])

    # deduplicate keeping order (simple)
    out_df = out_df.drop_duplicates().reset_index(drop=True)

    # shuffle
    out_df = out_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # split
    test_size = args.test_count if args.test_count else int(0.16 * len(out_df))
    train_df = out_df.iloc[:-test_size]
    test_df = out_df.iloc[-test_size:]

    train_df.to_csv('data/train_en.csv', index=False, encoding='utf-8')
    test_df.to_csv('data/test_en.csv', index=False, encoding='utf-8')

    # char map
    all_strings = list(out_df["input"].astype(str)) + list(out_df["target"].astype(str))
    chars = sorted(set("".join(all_strings)))
    char_map = {c: i+1 for i, c in enumerate(chars)}
    with open('python/artifacts/char_map_en.json', 'w', encoding='utf-8') as f:
        json.dump(char_map, f, ensure_ascii=False, indent=2)

    # labels
    labels = sorted(out_df["target"].unique())
    with open('python/labels_en.txt', 'w', encoding='utf-8') as f:
        for w in labels:
            f.write(w + "\n")

    print(f"Prepared train_en.csv ({len(train_df)}) and test_en.csv ({len(test_df)}) and wrote artifacts.") 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_csv', default='../data/en_kataho_code.csv',
                        help='CSV containing label, prefix_* and other columns')
    parser.add_argument('--test_count', type=int, default=None)
    args = parser.parse_args()
    main(args)
