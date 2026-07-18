import re
from difflib import get_close_matches

ITEM_SYNONYMS = {
    "espresso machine": ["espresso machine", "espresso maker", "coffee machine", "coffee maker",
                          "espresso", "coffee", "coffeemaker", "espressomachine", "coffee brewer",
                          "coffee pot", "coffee press", "capsule machine", "pod machine"],
    "robot vacuum": ["robot vacuum", "robotic vacuum", "vacuum cleaner", "vacuum", "roomba",
                      "robovac", "floor cleaner", "robot cleaner", "smart vacuum",
                      "autonomous vacuum", "vacuum robot", "cleaning robot", "floor robot",
                      "mop robot", "robot mop", "mopping robot", "vacuum mop",
                      "cleaning bot", "smart mop"],
    "air fryer": ["air fryer", "airfryer", "air-fryer", "fryer", "hot air fryer", "oil free fryer"],
    "air purifier": ["air purifier", "air cleaner", "purifier", "hepa purifier", "air filtration",
                      "air filter device"],
    "dishwasher": ["dishwasher", "dish washer", "dish cleaning machine"],
    "washing machine": ["washing machine", "washer", "laundry machine", "clothes washer"],
    "microwave": ["microwave", "microwave oven"],
    "blender": ["blender", "mixer", "food processor", "hand blender", "smoothie maker"],
    "kettle": ["kettle"],
}

FEATURE_SYNONYMS = {
    "quiet operation": ["quiet", "low noise", "less noise", "silent", "noiseless", "not loud",
                         "no noise", "whisper quiet", "sound reducing", "low decibel"],
    "low cost": ["low cost", "cheap", "affordable", "budget", "inexpensive", "low price",
                 "cost effective", "value for money", "wallet friendly"],
    "easy maintenance": ["easy maintenance", "low maintenance", "easy to clean", "easy cleaning",
                          "minimal maintenance", "hassle free", "no maintenance",
                          "self cleaning", "auto clean", "simple upkeep", "low upkeep"],
    "compact size": ["small", "compact", "space saving", "for small houses", "small apartment",
                      "tiny", "small kitchen", "slim design", "narrow", "small footprint"],
    "large capacity": ["big", "large", "for big houses", "large family", "large capacity",
                        "spacious", "big house", "high capacity", "extra large", "family size"],
    "energy efficient": ["energy efficient", "energy saving", "low power", "eco friendly",
                          "saves electricity", "low electricity bill", "power saving",
                          "green appliance", "reduced consumption"],
    "fast performance": ["fast", "quick", "rapid", "speedy", "cooks fast", "cleans fast",
                          "high speed", "time saving", "efficient cooking"],
    "smart features": ["smart", "app control", "wifi", "programmable", "timer", "scheduled",
                        "automatic", "voice control", "connected", "smart home", "remote control",
                        "app controlled", "bluetooth"],
    "durability": ["durable", "long lasting", "sturdy", "reliable", "well built", "robust",
                    "heavy duty", "built to last"],
    "surface adaptability": ["floor adaptable", "multi surface", "adapts to different floors",
                              "works on carpet and hardwood", "adjusts to floor type",
                              "floor sensing", "surface detection", "carpet and hard floor",
                              "all floor types", "cross surface"],
    "liquid resistance": ["liquid resistant", "liquid resistance", "waterproof", "water resistant",
                           "spill proof", "spill resistant", "leak proof", "wet dry",
                           "mop function", "wet cleaning", "moisture resistant",
                           "avoids liquid", "liquid detection", "handles spills"],
    "safety features": ["child safe", "childproof", "auto shutoff", "safety lock",
                         "overheat protection", "tip over protection"],
    "kitchen": ["kitchen"],
}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[’']", "'", text)
    text = text.replace("-", " ")  # split hyphenated compounds like "floor-adaptable"
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_matches(text: str, synonym_dict: dict, fuzzy_cutoff: float = 0.82):
    matched_keys = set()
    words = text.split()
    n = len(words)

    for key, phrases in synonym_dict.items():
        for phrase in phrases:
            phrase_norm = phrase.lower()
            phrase_len = len(phrase_norm.split())

            if re.search(r"\b" + re.escape(phrase_norm) + r"\b", text):
                matched_keys.add(key)
                break

            if phrase_len <= n:
                matched = False
                for i in range(n - phrase_len + 1):
                    window = " ".join(words[i:i + phrase_len])
                    if get_close_matches(phrase_norm, [window], n=1, cutoff=fuzzy_cutoff):
                        matched_keys.add(key)
                        matched = True
                        break
                if matched:
                    break

    return matched_keys


def parse_prompt(raw_text: str):
    """
    Parses free-text input from the UI text box.
    Returns a list of matched keywords (item + features),
    or a string message if no appliance item is found.
    """
    if not raw_text or not raw_text.strip():
        return "No text provided. Please describe the appliance and desired features."

    text = _normalize(raw_text)

    matched_items = _find_matches(text, ITEM_SYNONYMS)
    matched_features = _find_matches(text, FEATURE_SYNONYMS)

    if not matched_items:
        return (
            "Could not identify a home appliance in your text. "
            "Try mentioning a product like 'espresso machine', 'robot vacuum', "
            "'air fryer', or 'air purifier'."
        )

    keywords = sorted(matched_items) + sorted(matched_features)
    return keywords