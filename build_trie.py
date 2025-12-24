# build_trie.py
import os
import json
import pandas as pd

def insert(root, word):
    """Insert a word into the trie."""
    node = root
    for ch in word:
        node = node.setdefault(ch, {})
    node['_end'] = True

def build_trie(words):
    """Build a trie from a list of words."""
    root = {}
    for w in words:
        insert(root, w)
    return root

def main(csv_path='../data/training_samples_full.csv', artifacts_dir='python/artifacts'):
    # Load CSV
    df = pd.read_csv(csv_path, encoding='utf-8')

    if 'target' not in df.columns:
        raise ValueError("CSV must contain a 'target' column")

    # Extract vocabulary (strip and remove NaNs)
    vocab = df['target'].dropna().astype(str).str.strip().unique().tolist()

    # Build trie
    trie = build_trie(vocab)

    # Ensure artifacts folder exists
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save trie.json
    trie_path = os.path.join(artifacts_dir, 'trie.json')
    with open(trie_path, 'w', encoding='utf-8') as f:
        json.dump(trie, f, ensure_ascii=False, indent=2)

    # Save labels.txt
    labels_path = os.path.join('python', 'labels.txt')
    with open(labels_path, 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write(w + '\n')

    print(f"Saved trie.json with {len(vocab)} words")
    print(f"Saved labels.txt with {len(vocab)} words")

if __name__ == '__main__':
    main()

