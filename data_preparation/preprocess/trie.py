# build_trie.py
import json
import os
import pandas as pd
from config import TRAIN_CSV

def insert(root, word):
    """Insert a word into the trie."""
    node = root
    for ch in word:
        if ch not in node:
            node[ch] = {}
        node = node[ch]
    node['_end'] = True

def main():
    # Load CSV
    df = pd.read_csv(TRAIN_CSV, encoding = 'utf-8')

    # Extract correct words
    vocab = df['correct_word'].dropna().astype(str).str.strip().tolist()

    # Build trie
    root = {}
    for w in vocab:
        insert(root, w)

    # Resolve artifacts dir relative to this script's location (data_preparation/artifacts)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, 'data_preparation', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save trie
    with open(os.path.join(artifacts_dir, 'trie.json'), 'w', encoding='utf-8') as f:
        json.dump(root, f, ensure_ascii=False, indent=2)

    # Save labels (words)
    with open(os.path.join(artifacts_dir, 'labels.txt'), 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write(w + '\n')

    print('Saved trie.json with', len(vocab), 'words')

if __name__ == '__main__':
    main()