class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def _dfs(self, node, prefix, res):
        if node.is_word:
            res.append(prefix)
        for ch, child in node.children.items():
            self._dfs(child, prefix + ch, res)

    def search_prefix(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        res = []
        self._dfs(node, prefix, res)
        return res
