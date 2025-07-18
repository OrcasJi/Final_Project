
from fuzzywuzzy import fuzz

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

def fuzzy_match_style(text):
    text = text.lower()
    for style, synonyms in STYLE_KEYWORDS.items():
        for word in synonyms:
            if fuzz.partial_ratio(word, text) >= 80:
                return style
    return None

def fuzzy_match_category(text):
    text = text.lower()
    for cat, synonyms in CATEGORY_KEYWORDS.items():
        for word in synonyms:
            if fuzz.partial_ratio(word, text) >= 80:
                return cat
    return None
