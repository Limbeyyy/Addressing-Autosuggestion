# prepare_data/morpheme_generator.py
"""
Morpheme-based prefix/suffix generator:
- truncates the correct word at morpheme boundaries (simple heuristic)
- reattaches common suffixes or removes them to create plausible prefixes
"""
from prepare_data.utils import RNG

COMMON_SUFFIXES = ["ता", "हरु", "हरु", "न्", "िय", "ो", "ा", "ि", "ी"]

def generate_prefix_variants(word, max_variants=4):
    variants = set()
    # naive morpheme cut: split into syllable-like chunks (every 2 chars)
    step = max(1, len(word)//3)
    for i in range(1, len(word), step):
        v = word[:i]
        if v and v != word:
            variants.add(v)
        if len(variants) >= max_variants:
            break
    # attach common suffixes
    for suf in COMMON_SUFFIXES:
        if len(variants) >= max_variants:
            break
        if not word.endswith(suf):
            variants.add(word + suf)
    return list(variants)

