# prepare_data/prepare_data.py
import os
import json
import argparse
import pandas as pd

from prepare_data.utils import parse_list
from prepare_data.generator_rules import apply_rule_by_name, double_letter, swap_adjacent
from prepare_data.phonetic_rules import phonetic_variants
from prepare_data.keyboard_rules import keyboard_neighbors
from prepare_data.neural_synthetic import generate_neural_like
from prepare_data.morpheme_generator import generate_prefix_variants

# ensure reproducible
import random
random.seed(42)

def expand_row(row):
    """
    Given a CSV row with 'correct_word' etc., return list of (input, target) tuples.
    """
    correct = str(row.get("correct_word", "")).strip()
    if not correct:
        return []

    outs = set()

    # 1) include prefixes (prefix_2..prefix_7)
    for c in row.index:
        if str(c).startswith("prefix_"):
            v = str(row[c]).strip()
            if v and v.lower() != "nan":
                outs.add((v, correct))

    # 2) include correct word itself
    outs.add((correct, correct))

    # 3) include common_typos column
    for t in parse_list(row.get("common_typos", "[]")):
        t = str(t).strip()
        if t:
            outs.add((t, correct))

    # 4) apply explicit typo_patterns rules (like 'ष→श', 'add_halant', 'consonant_swap' etc.)
    typo_patterns = parse_list(row.get("typo_patterns", "[]"))
    for rule in typo_patterns:
        if not rule:
            continue
        # some rules may be composite or descriptive; try apply
        try:
            gen = apply_rule_by_name(correct, rule)
            if gen:
                outs.add((gen, correct))
        except Exception:
            pass

    # 5) apply correction_rules and phonetic_rules to generate candidates
    phonetic_rules = parse_list(row.get("phonetic_rules", "[]"))
    # convert phonetic rules maybe like ["ʃ→ʂ"]
    # use phonetic_variants generator (it uses an internal map)
    pv = phonetic_variants(correct, max_variants=4) if hasattr(__import__('prepare_data.phonetic_rules'), 'phonetic_variants') else []
    for p in pv:
        outs.add((p, correct))

    # 6) keyboard neighbors
    kb = keyboard_neighbors(correct, max_variants=4)
    for k in kb:
        outs.add((k, correct))

    # 7) neural-like synthetic variations
    for n in generate_neural_like(correct, n=4):
        outs.add((n, correct))

    # 8) morpheme/prefix variants
    for p in generate_prefix_variants(correct, max_variants=4):
        outs.add((p, correct))

    # 9) small edits: double letters, swap
    outs.add((double_letter(correct), correct))
    outs.add((swap_adjacent(correct), correct))

    # filter empties and return
    return [(i, t) for i, t in outs if i and str(i).strip()]

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

    train_df.to_csv('data/train.csv', index=False, encoding='utf-8')
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8')

    # char map
    all_strings = list(out_df["input"].astype(str)) + list(out_df["target"].astype(str))
    chars = sorted(set("".join(all_strings)))
    char_map = {c: i+1 for i, c in enumerate(chars)}
    with open('python/artifacts/char_map.json', 'w', encoding='utf-8') as f:
        json.dump(char_map, f, ensure_ascii=False, indent=2)

    # labels
    labels = sorted(out_df["target"].unique())
    with open('python/labels.txt', 'w', encoding='utf-8') as f:
        for w in labels:
            f.write(w + "\n")

    print(f"Prepared train.csv ({len(train_df)}) and test.csv ({len(test_df)}) and wrote artifacts.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_csv', default='../data/training_samples_full.csv',
                        help='CSV containing correct_word, prefix_* and other columns')
    parser.add_argument('--test_count', type=int, default=None)
    args = parser.parse_args()
    main(args)
