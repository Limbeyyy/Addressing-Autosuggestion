# prepare_data/utils.py
import ast
import random

RNG = random.Random(42)

def parse_list(x):
    """Safely parse lists stored as strings in CSV columns."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        # fallback: comma separated
        s = str(x)
        if not s or s.lower() == "nan":
            return []
        return [p.strip() for p in s.split(",") if p.strip()]

def sample_with_prob(lst, p=0.5):
    """Return subset sampled from lst with probability p each item."""
    return [x for x in lst if RNG.random() < p]

