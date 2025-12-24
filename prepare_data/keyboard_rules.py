# prepare_data/keyboard_rules.py
"""
Keyboard-nearby typo generator.
We create a small adjacency map for Devanagari keyboard (approximate) and QWERTY Latin
so we can simulate mistypes (for transliteration and romanized inputs).
"""
from prepare_data.utils import RNG

# Very small/nebulous neighbor map for demonstration.
# Extend this map depending on the real keyboard layout you expect.
NEIGHBORS = {
    "क": ["ख", "ग"],
    "ख": ["क", "ग"],
    "ग": ["घ", "क"],
    "घ": ["ग"],
    "श": ["ष", "स"],
    "स": ["श"],
    "ा": ["ि", "े"],
    "ि": ["ा", "ी"],
    "र": ["र"],  # placeholder
    # Latin examples
    "a": ["s", "q"],
    "s": ["a", "d"],
    "d": ["s", "f"],
}

def keyboard_neighbors(word, max_variants=4):
    variants = set()
    for i, ch in enumerate(word):
        if ch in NEIGHBORS:
            for nb in NEIGHBORS[ch]:
                v = word[:i] + nb + word[i+1:]
                variants.add(v)
                if len(variants) >= max_variants:
                    return list(variants)
    return list(variants)

