import json
import pandas as pd

df = pd.read_csv("prepare_data/data/train.csv", encoding="utf-8")
df["input"] = df["input"].str.lower().str.strip()
df["target"] = df["target"].str.strip()

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.nepali = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, nep):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_word = True
        node.nepali = nep

    def to_dict(self, node=None):
        if node is None:
            node = self.root
        d = {}
        for c, child in node.children.items():
            d[c] = self.to_dict(child)
        if node.is_word:
            d["_end"] = True
            d["_nep"] = node.nepali
        return d


trie = Trie()
for e, n in zip(df["input"], df["target"]):
    trie.insert(e, n)

trie_dict = trie.to_dict()

# Pretty JSON tree
with open("english_trie.json", "w", encoding="utf-8") as f:
    json.dump(trie_dict, f, ensure_ascii=False, indent=2)

print("Tree JSON saved!")
