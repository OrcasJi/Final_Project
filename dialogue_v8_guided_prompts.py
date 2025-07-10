
import joblib
import pandas as pd
import random
from fuzzy_match import fuzzy_match_style, fuzzy_match_category

model = joblib.load("expanded_intent_classifier_v4.pkl")
catalog = pd.read_csv("product_catalog_100_items.csv")

context = {"category": None, "style": None, "max_price": None}
awaiting_confirmation = False
last_recommendation_count = 0
invalid_count = 0

thank_you_responses = [
    "You're welcome! Let me know if you'd like to see more.",
    "No problem! Happy to help.",
    "You're very welcome. I'm here if you need anything else!"
]

confirmation_responses = [
    "Great! I'm glad you like it.",
    "Awesome choice! Hope it works well for you.",
    "Perfect! Let me know if there's anything else."
]

rejection_responses = [
    "No worries! Let's try something else.",
    "Okay, we can explore more options.",
    "Understood! I'll find other items you might like."
]

done_responses = [
    "Thank you for shopping with us! Have a great day!",
    "It was a pleasure helping you. Goodbye!",
    "Hope to see you again soon! Take care."
]

guidance_responses = [
    "That's totally okay! A lot of people feel the same.",
    "No problem! I can help guide you.",
    "Let's start somewhere — I’ll suggest some good picks."
]

intent_responses = {
    "ask_product": [
        "What type of product are you looking for? Jackets, shoes, or something else?"
    ],
    "ask_style": [
        "What style are you into? Casual, formal, sporty?"
    ],
    "ask_price": [
        "What's your budget? Please enter a maximum price (e.g., under 100)."
    ],
    "greet": [
        "Hello! How can I assist you with shopping today?"
    ],
    "goodbye": done_responses
}

def extract_price(text):
    import re
    matches = re.findall(r"\d+", text)
    if matches:
        return int(matches[0])
    return None

def extract_category(text):
    keywords = ["jacket", "shoes", "shirt", "pants", "dress"]
    for kw in keywords:
        if kw in text.lower():
            return kw
    return None

def extract_style(text):
    keywords = ["casual", "formal", "sporty"]
    for kw in keywords:
        if kw in text.lower():
            return kw
    return None

def prompt_missing_context():
    prompts = []
    if not context["category"]:
        prompts += intent_responses["ask_product"]
    if not context["style"]:
        prompts += intent_responses["ask_style"]
    if not context["max_price"]:
        prompts += intent_responses["ask_price"]
    return prompts

def suggest_remaining_context():
    hints = []
    if not context["category"]:
        hints.append("what category are you looking for?")
    if not context["style"]:
        hints.append("any preferred style?")
    if not context["max_price"]:
        hints.append("what's your budget?")
    if hints:
        return " You can tell me more — " + " ".join(hints)
    return ""

def recommend_products(partial_ok=True):
    global last_recommendation_count
    df = catalog.copy()
    if context["category"]:
        df = df[df["category"].str.lower() == context["category"].lower()]
    if context["style"]:
        df = df[df["style"].str.lower() == context["style"].lower()]
    if context["max_price"]:
        df = df[df["price"] <= context["max_price"]]
    if not partial_ok and df.empty:
        return "Sorry, I couldn't find anything matching all your preferences."
    elif df.empty:
        df = catalog.sample(3)
    else:
        df = df.sample(min(3, len(df)))
    last_recommendation_count = len(df)

    intro = "Here are some suggestions"
    if context["style"]:
        intro += f" based on style: {context['style']}"
    elif context["category"]:
        intro += f" based on category: {context['category']}"
    elif context["max_price"]:
        intro += f" under £{context['max_price']}"
    intro += "."
    recs = "\n".join([f"- {row['name']} (£{row['price']:.2f})" for _, row in df.iterrows()])
    return f"{intro}\n{recs}"

def reset_context():
    global context, awaiting_confirmation
    context = {"category": None, "style": None, "max_price": None}
    awaiting_confirmation = False

print("🛍️ AI Shopping Assistant: Hi there! Ask me anything about fashion or shopping.")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    predicted_intent = model.predict([user_input])[0]

    if predicted_intent == "done":
        print("Assistant:", random.choice(done_responses))
        break

    if predicted_intent == "thank_you":
        print("Assistant:", random.choice(thank_you_responses))
        continue

    if predicted_intent == "greet":
        print("Assistant:", random.choice(intent_responses["greet"]))
        continue

    if predicted_intent == "confirmation" and awaiting_confirmation and last_recommendation_count > 0:
        print("Assistant:", random.choice(confirmation_responses))
        print("Assistant:", random.choice(intent_responses["ask_product"]))
        reset_context()
        continue

    if predicted_intent == "reject" and awaiting_confirmation:
        print("Assistant:", random.choice(rejection_responses))
        reset_context()
        continue

    if predicted_intent == "undecided":
        print("Assistant:", random.choice(guidance_responses))
        print("Assistant:", recommend_products() + suggest_remaining_context())
        reset_context()
        continue

    price = extract_price(user_input)
    style = extract_style(user_input) or fuzzy_match_style(user_input)
    category = extract_category(user_input) or fuzzy_match_category(user_input)

    extracted_info = False
    if price:
        context["max_price"] = price
        extracted_info = True
    if style:
        context["style"] = style
        extracted_info = True
    if category:
        context["category"] = category
        extracted_info = True

    print("DEBUG:", context)

    ready_for_full_recommendation = all([
        context["category"],
        context["style"],
        context["max_price"]
    ])

    if context["category"] or context["style"] or context["max_price"]:
        print("Assistant:", recommend_products(partial_ok=True) + suggest_remaining_context())
        if last_recommendation_count <= 3 and ready_for_full_recommendation:
            print("Assistant: Do any of these options look good to you?")
            awaiting_confirmation = True
        continue

    if not extracted_info:
        invalid_count += 1
        if invalid_count >= 3:
            print("Assistant: I can help better if you share your style, product type, or budget 😊")
            invalid_count = 0
    else:
        invalid_count = 0

    missing = prompt_missing_context()
    if missing:
        print("Assistant:", random.choice(missing))
