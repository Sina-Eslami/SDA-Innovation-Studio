from news_ingest import NewsClient

from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from pathlib import Path

import pandas as pd
import numpy as np

import re

#-------making keyword related dataset---------------------------

keyword_sets = {
    "espresso_machine": [
        "espresso machine",
        "coffee maker",
        "home espresso",
        "automatic espresso machine"
    ],

    "robot_vacuum": [
        "robot vacuum",
        "robotic vacuum cleaner",
        "smart vacuum",
        "autonomous cleaning robot"
    ],

    "air_fryer": [
        "air fryer",
        "smart air fryer",
        "oil free fryer"
    ],

    "air_purifier": [
        "air purifier",
        "home air filtration",
        "HEPA air purifier"
    ],

    "dishwasher": [
        "dishwasher",
        "smart dishwasher",
        "energy efficient dishwasher"
    ],

    "washing_machine": [
        "washing machine",
        "smart washer",
        "energy efficient washing machine"
    ],

    "microwave": [
        "microwave oven",
        "smart microwave",
        "convection microwave"
    ],

    "blender": [
        "blender",
        "high performance blender",
        "smart blender"
    ],

    # Product attributes
    "quiet_operation": [
        "quiet appliance",
        "silent motor technology",
        "low noise appliance",
        "noise reduction technology"
    ],

    "low_cost": [
        "affordable appliance",
        "budget friendly appliance",
        "cheap smart appliance",
        "value appliance"
    ],

    "easy_maintenance": [
        "easy to clean appliance",
        "low maintenance appliance",
        "self cleaning appliance"
    ],

    "compact_size": [
        "compact appliance",
        "small kitchen appliance",
        "space saving appliance"
    ],

    "large_capacity": [
        "large capacity appliance",
        "high capacity washer",
        "extra large appliance"
    ],

    "energy_efficient": [
        "energy efficient appliance",
        "low energy consumption appliance",
        "energy saving technology"
    ],

    "fast_performance": [
        "fast cooking appliance",
        "quick performance appliance",
        "rapid heating technology"
    ],

    "smart_features": [
        "smart appliance",
        "connected home appliance",
        "AI powered appliance",
        "app controlled appliance"
    ],

    "durability": [
        "durable appliance",
        "long lasting appliance",
        "reliable appliance",
        "product lifespan"
    ],

    "surface_adaptability": [
        "multi surface cleaning",
        "works on different surfaces",
        "floor compatibility"
    ],

    "liquid_resistance": [
        "water resistant appliance",
        "spill resistant appliance",
        "waterproof technology"
    ],

    "safety_features": [
        "appliance safety features",
        "child safety appliance",
        "automatic shutoff technology"
    ]
}

news_frames = []

for category, keywords in keyword_sets.items():

    for keyword in keywords:

        try:
            client = NewsClient(default_keywords=[keyword])

            # On a real version year_back should be 5
            df = client.fetch_last_n_years(years_back=0.08)

            df["category"] = category
            df["keyword"] = keyword

            news_frames.append(df)
        except Exception as e:
            print('You hit your NEWS api limit.')
            print(e)
            df = pd.DataFrame()

if news_frames != []:
    news_df = pd.concat(
        news_frames,
        ignore_index=True
    )

else:
    print('No news has be gathered from api.')

BASE_DIR = Path(__file__).resolve().parent
output_dir = BASE_DIR / ".." / ".." / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)
filename = f"raw-news.csv"
df.to_csv(output_dir / filename, index=False)
print(f"File {output_dir / filename} saved.")

#-------cleaning-------------------------------------------------

def clean_news_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_cols = ["source_name", "author", "title", "description", "content"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            df[col] = df[col].replace({"": np.nan, "None": np.nan, "nan": np.nan})

    df["source_name_norm"] = df["source_name"].str.lower()
    df["author_norm"] = df["author"].str.lower()

    def clean_url(u):
        if pd.isna(u):
            return np.nan
        try:
            parsed = urlparse(u)
            q = [
                (k, v) for k, v in parse_qsl(parsed.query)
                if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}
            ]
            new_query = urlencode(q)
            return urlunparse((
                parsed.scheme, parsed.netloc, parsed.path.rstrip("/"),
                parsed.params, new_query, ""
            ))
        except Exception:
            return u

    df["url"] = df["url"].apply(clean_url)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    def normalize_keywords(val):
        if isinstance(val, list):
            return sorted(set(str(k).strip().lower() for k in val if str(k).strip()))
        if pd.isna(val):
            return []
        s = str(val).strip().strip("[]")
        parts = re.split(r"[,;]", s)
        return sorted(set(p.strip().strip("'\"").lower() for p in parts if p.strip()))

    df["matched_keywords"] = df["matched_keywords"].apply(normalize_keywords)
    df["n_matched_keywords"] = df["matched_keywords"].apply(len)

    df = df[~(df["title"].isna() & df["content"].isna())]

    df = df.dropna(subset=["published_at"])

    df = df.drop_duplicates(subset=["url"], keep="first")

    df = df.drop_duplicates(subset=["title", "source_name_norm"], keep="first")

    df["author"] = df["author"].fillna("Unknown")
    df["source_name"] = df["source_name"].fillna("Unknown")
    df["description"] = df["description"].fillna("")
    df["content"] = df["content"].fillna("")

    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    return df

#-------feature engineering------------------------------------------

def engineer_news_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for news_df, assumes clean_news_df() already ran.
    Skips: whitespace normalization, NaN filling, published_at parsing, dedup
    (all handled by clean_news_df already).
    """
    df = df.copy()

    def strip_html_urls_boilerplate(text):
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\[\+\d+\s*chars\]", " ", text)  # NewsAPI truncation artifact
        return re.sub(r"\s+", " ", text).strip()

    for col in ["title", "description", "content"]:
        df[f"{col}_clean"] = df[col].apply(strip_html_urls_boilerplate)

    def combine_texts(row):
        parts, seen = [], set()
        for col in ["title_clean", "description_clean", "content_clean"]:
            t = row[col].strip()
            key = t.lower()
            if t and key not in seen:
                parts.append(t)
                seen.add(key)
        return " ".join(parts)

    df["full_text"] = df.apply(combine_texts, axis=1)

    for col in ["title_clean", "description_clean", "content_clean", "full_text"]:
        df[f"{col}_word_count"] = df[col].str.split().apply(len)

    df["title_exclaim_count"] = df["title_clean"].str.count("!")
    df["title_is_upper_ratio"] = df["title_clean"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    df["content_truncated"] = df["content"].str.contains(r"\[\+\d+\s*chars\]", regex=True)
    df["has_description"] = df["description_clean"].str.len() > 0
    df["has_content"] = df["content_clean"].str.len() > 0
    df["text_source_quality"] = np.select(
        [df["has_content"] & df["has_description"],
         df["has_description"] & ~df["has_content"],
         ~df["has_description"] & ~df["has_content"]],
        ["full", "partial", "title_only"]
    )

    df["pub_year"] = df["published_at"].dt.year
    df["pub_month"] = df["published_at"].dt.month
    df["pub_week"] = df["published_at"].dt.isocalendar().week
    df["pub_dayofweek"] = df["published_at"].dt.day_name()

    return df