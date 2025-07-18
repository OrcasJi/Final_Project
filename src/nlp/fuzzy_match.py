
"""Lightweight fuzzy matching utilities used by the dialogue system.

This module originally depended on the external ``fuzzywuzzy`` package.  The
environment used for running the assistant does not always have that
dependency available and we cannot install it without internet access.  To keep
the code self‑contained we implement a very small subset of the functionality
that we need here.
"""

from difflib import SequenceMatcher


def _partial_ratio(a: str, b: str) -> int:
    """Return a similarity score between 0 and 100 for substrings.

    This is a tiny replacement for :func:`fuzzywuzzy.fuzz.partial_ratio`.  It
    looks for the best matching substring of ``b`` against ``a`` and returns the
    match ratio as a percentage.
    """

    if not a or not b:
        return 0

    a = a.lower()
    b = b.lower()
    if len(a) > len(b):
        a, b = b, a

    max_ratio = 0.0
    len_a = len(a)
    for i in range(len(b) - len_a + 1):
        sub = b[i : i + len_a]
        ratio = SequenceMatcher(None, a, sub).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
        if max_ratio == 1.0:
            break

    return int(round(max_ratio * 100))

STYLE_KEYWORDS = {
    "casual": [
        "relaxed", "laid back", "easygoing", "simple", "streetwear",
        "chill", "basic", "everyday", "regular", "loose", "cool", "lowkey"
    ],
    "formal": [
        "elegant", "dressy", "suit", "professional", "office", "ceremony",
        "business", "classic", "gala", "fancy", "neat", "well dressed"
    ],
    "sporty": [
        "athletic", "active", "gym", "fit", "training", "sportswear", "jogging",
        "outdoor", "movement", "performance", "track", "fitness", "sport"
    ]
}

CATEGORY_KEYWORDS = {
    "jacket": [
        "hoodie", "coat", "blazer", "parka", "windbreaker", "bomber", "zip up",
        "outerwear", "puffer", "down jacket", "anorak", "trench"
    ],
    "shoes": [
        "sneakers", "trainers", "boots", "footwear", "sandals", "heels",
        "loafers", "derby", "canvas", "leather shoes", "slip-ons", "kicks"
    ],
    "shirt": [
        "t-shirt", "tee", "polo", "top", "blouse", "button up", "long sleeve",
        "casual shirt", "crop top", "tank top", "collared shirt", "jersey"
    ],
    "pants": [
        "jeans", "trousers", "leggings", "joggers", "cargo", "chinos",
        "sweatpants", "slacks", "bottoms", "denim", "formal pants"
    ],
    "dress": [
        "gown", "partywear", "one piece", "evening dress", "maxi", "midi",
        "mini dress", "formalwear", "summer dress", "cocktail", "bodycon"
    ]
}

def fuzzy_match_style(text: str) -> str | None:
    """Attempt to infer a style from ``text`` using fuzzy matching."""

    text = text.lower()
    for style, synonyms in STYLE_KEYWORDS.items():
        for word in synonyms:
            if _partial_ratio(word, text) >= 80:
                return style
    return None

def fuzzy_match_category(text: str) -> str | None:
    """Attempt to infer a product category from ``text``."""

    text = text.lower()
    for cat, synonyms in CATEGORY_KEYWORDS.items():
        for word in synonyms:
            if _partial_ratio(word, text) >= 80:
                return cat
    return None
