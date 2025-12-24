# prepare_data/generator_rules.py
from prepare_data.utils import RNG
import re

# Basic substitution and halant helpers
def apply_substitution(word, src, dst):
    return word.replace(src, dst)

def add_halant(word):
    # insert halant before last consonant cluster if applicable
    if not word:
        return word
    # naive: add halant before last character
    return word[:-1] + "्" + word[-1]

def remove_halant(word):
    return word.replace("्", "")

def double_letter(word):
    if len(word) < 2:
        return word
    # pick a random position and duplicate char
    i = RNG.randrange(len(word))
    return word[:i] + word[i] + word[i] + word[i+1:]

def delete_char(word):
    if len(word) <= 1:
        return word
    i = RNG.randrange(len(word))
    return word[:i] + word[i+1:]

def swap_adjacent(word):
    if len(word) < 2:
        return word
    i = RNG.randrange(len(word)-1)
    return word[:i] + word[i+1] + word[i] + word[i+2:]

# apply a named rule (simple)
def apply_rule_by_name(word, rule_name):
    """
    Recognizes:
      'add_halant', 'remove_halant', 'double', 'delete', 'swap'
      or substitutions like 'ष→श'
    """
    if not rule_name:
        return None
    if "→" in rule_name:
        src, dst = rule_name.split("→")
        return apply_substitution(word, src, dst)
    rn = rule_name.strip().lower()
    if rn == "add_halant":
        return add_halant(word)
    if rn == "remove_halant":
        return remove_halant(word)
    if rn in ("double", "double_letter"):
        return double_letter(word)
    if rn in ("delete", "delete_char"):
        return delete_char(word)
    if rn in ("swap", "swap_adjacent"):
        return swap_adjacent(word)
    # unknown rule
    return None

