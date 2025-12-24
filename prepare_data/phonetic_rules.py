# prepare_data/phonetic_rules.py
"""
Phonetic confusion generator for Nepali/Hindi-like mappings.
This file contains common phonetic substitution pairs and a generator
that applies them to produce plausible confusions.
"""
from prepare_data.utils import RNG

# Example mapping: each key -> list of plausible confusions
PHONETIC_MAP = {
    "श": ["ष", "स"],
    "ष": ["श", "स"],
    "स": ["श"],
    "ब": ["भ"],
    "भ": ["ब"],
    "द": ["ध"],
    "ध": ["द"],
    "ग": ["घ"],
    "घ": ["ग"],
    "क": ["ख"],
    "ख": ["क"],
    "च": ["छ"],
    "छ": ["च"],
    # vowels / matras (examples)
    "ि": ["ी"],  # short i -> long i
    "ा": ["ि"],
    "े": ["ै"],
    "ो": ["ौ"],
    # retroflex-flap confusions (simple)
    "र": ["ऱ"]  # in some orthographies
}

def phonetic_variants(word, max_variants=3):
    variants = set()
    for i, ch in enumerate(word):
        if ch in PHONETIC_MAP:
            for alt in PHONETIC_MAP[ch]:
                v = word[:i] + alt + word[i+1:]
                if v != word:
                    variants.add(v)
                    if len(variants) >= max_variants:
                        return list(variants)
    # small chance to apply multiple changes
    if len(variants) < max_variants and RNG.random() < 0.2:
        # attempt two substitutions
        chars = list(word)
        changed = False
        for i, ch in enumerate(chars):
            if ch in PHONETIC_MAP and RNG.random() < 0.5:
                chars[i] = PHONETIC_MAP[ch][0]
                changed = True
        if changed:
            variants.add("".join(chars))
    return list(variants)

