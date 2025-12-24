# prepare_data/neural_synthetic.py
"""
Lightweight 'neural-style' synthetic typo generator that:
- uses weighted random edits (insert/delete/substitute/transpose)
- no heavy ML models; just probabilistic edits to diversify data
"""
from prepare_data.utils import RNG
import random
import itertools

VOWELS = "अआइईउऊएऐओऔॅॅॉ"
CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"

def random_insert(word):
    # insert a random character (vowel or consonant)
    pool = VOWELS + CONSONANTS
    ch = RNG.choice(pool)
    i = RNG.randrange(len(word)+1)
    return word[:i] + ch + word[i:]

def random_substitute(word):
    if not word:
        return word
    i = RNG.randrange(len(word))
    pool = VOWELS + CONSONANTS
    ch = RNG.choice(pool)
    return word[:i] + ch + word[i+1:]

def random_transpose(word):
    if len(word) < 2:
        return word
    i = RNG.randrange(len(word)-1)
    lst = list(word)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst)

def generate_neural_like(word, n=3):
    variants = set()
    ops = [random_insert, random_substitute, random_transpose]
    for _ in range(n):
        w = word
        # apply 1-2 ops
        k = RNG.choice([1,2])
        for __ in range(k):
            op = RNG.choice(ops)
            w = op(w)
        if w != word:
            variants.add(w)
        if len(variants) >= n:
            break
    return list(variants)

