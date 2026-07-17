from patents_ingest import PatentBiblioClient

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

def collect_patent_df():

    patent_frames = []

    for category, keywords in keyword_sets.items():
        for keyword in keywords:
            try:
                patent_kw = PatentBiblioClient(
                    default_keywords=[keyword],
                    save=False
                )

                df = patent_kw.fetch_biblio_last_n_years(years_back=1)

                if df is None or df.empty:
                    continue

                df["category"] = category
                df["keyword"] = keyword

                patent_frames.append(df)

            except Exception as e:
                print(f"Could not fetch patents for '{keyword}'.")
                print(e)

    if patent_frames:
        patent_df = pd.concat(patent_frames, ignore_index=True)

        BASE_DIR = Path(__file__).resolve().parent
        output_dir = BASE_DIR / ".." / ".." / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = "raw-patent.csv"
        output_path = output_dir / filename

        patent_df.to_csv(output_path, index=False)

        print(f"Saved {len(patent_df)} patents to {output_path}")

    else:
        print("No patents were gathered from the API.")
        patent_df = pd.DataFrame()

#-------cleaning-------------------------------------------------

def clean_patents_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["family_id", "country", "doc_number", "kind"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()
            df[col] = df[col].replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})

    for col in ["title", "abstract"]:
        df[col] = (
            df[col].astype("string").str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        df[col] = df[col].replace({"": np.nan, "None": np.nan, "nan": np.nan})

    s = df["publication_date"].astype("string").str.strip()

    parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    still_missing = parsed.isna() & s.notna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(
            s.loc[still_missing], errors="coerce", format="mixed"
        )

    df["publication_date"] = parsed

    def normalize_keywords(val):
        if isinstance(val, list):
            return sorted(set(str(k).strip().lower() for k in val if str(k).strip()))
        if pd.isna(val):
            return []
        s = str(val).strip().strip("[]")
        parts = re.split(r"[,;]", s)
        return sorted(set(p.strip().strip("'\"").lower() for p in parts if p.strip()))

    df["keywords"] = df["keywords"].apply(normalize_keywords)
    df["n_keywords"] = df["keywords"].apply(len)

    df["doc_id"] = df["country"].fillna("") + df["doc_number"].fillna("") + df["kind"].fillna("")
    df = df.drop_duplicates(subset=["doc_id"], keep="first")

    df = df[~(df["title"].isna() & df["abstract"].isna())]
    df = df.dropna(subset=["publication_date"])

    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["family_id"] = df["family_id"].fillna("UNKNOWN")

    df = df.sort_values("publication_date", ascending=False).reset_index(drop=True)
    return df

#-------feature engineering------------------------------------------

def engineer_patents_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["full_text"] = (df["title"] + " " + df["abstract"]).str.strip()
    df["title_word_count"] = df["title"].str.split().apply(len)
    df["abstract_word_count"] = df["abstract"].str.split().apply(len)
    df["has_abstract"] = df["abstract"].str.len() > 0

    df["pub_year"] = df["publication_date"].dt.year
    df["pub_month"] = df["publication_date"].dt.month
    df["pub_quarter"] = df["publication_date"].dt.quarter

    kind_map = {
        "A": "application", "A1": "application", "A2": "application", "A9": "application",
        "B": "grant", "B1": "grant", "B2": "grant",
        "U": "utility_model", "S": "design",
    }
    df["kind_category"] = df["kind"].map(kind_map).fillna("other")

    family_counts = df.groupby("family_id")["doc_id"].transform("count")
    df["family_size"] = np.where(df["family_id"] == "UNKNOWN", 1, family_counts)

    df["is_first_in_family"] = (
        df.groupby("family_id")["publication_date"].rank(method="first", ascending=True) == 1
    )
    df.loc[df["family_id"] == "UNKNOWN", "is_first_in_family"] = True

    return df