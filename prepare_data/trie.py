# build_trie.py
import csv
import json
import os
import pandas as pd

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
    df = pd.read_csv(r'C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\prepare_data\data\kataho_indian_sheet.csv', encoding='utf-8')

    # Extract correct words
    vocab = df['correct_word'].dropna().astype(str).str.strip().tolist()

    # Build trie
    root = {}
    for w in vocab:
        insert(root, w)

    # Create artifacts folder
    os.makedirs('python/artifacts', exist_ok=True)

    # # Save trie
    # with open('python/artifacts/trie.json', 'w', encoding='utf-8') as f:
    #     json.dump(root, f, ensure_ascii=False, indent=2)

    # Save labels (words)
    with open('python/labels.txt', 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write(w + '\n')

    print('Saved trie.json with', len(vocab), 'words')

if __name__ == '__main__':
    main()