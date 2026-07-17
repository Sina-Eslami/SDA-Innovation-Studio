from src.data_builder.patents_ingest import PatentBiblioClient

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

    # Publication date: strip trailing ".0" artifact from float-dtype CSV columns before parsing
    pub_date_str = df["publication_date"].astype("string").str.strip()
    pub_date_str = pub_date_str.str.replace(r"\.0$", "", regex=True)

    parsed = pd.to_datetime(pub_date_str, format="%Y%m%d", errors="coerce")

    still_missing = parsed.isna() & pub_date_str.notna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(
            pub_date_str.loc[still_missing], errors="coerce", format="mixed"
        )

    df["publication_date"] = parsed

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

    required_cols = {"title", "abstract", "publication_date", "kind", "family_id", "doc_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"engineer_patents_features missing required columns: {missing_cols}")

    if df.empty:
        return df

    # Ensure publication_date is real datetime dtype, in case upstream changed
    if not pd.api.types.is_datetime64_any_dtype(df["publication_date"]):
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")

    # Text features
    df["full_text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.strip()
    df["title_word_count"] = df["title"].fillna("").str.split().apply(len)
    df["abstract_word_count"] = df["abstract"].fillna("").str.split().apply(len)
    df["has_abstract"] = df["abstract"].fillna("").str.len() > 0
    df["has_usable_text"] = df["full_text"].str.len() > 0

    # Date-derived features (guard against NaT rows)
    df["pub_year"] = df["publication_date"].dt.year
    df["pub_month"] = df["publication_date"].dt.month
    df["pub_quarter"] = df["publication_date"].dt.quarter

    # Broader kind-code mapping, normalized to uppercase, with prefix fallback
    kind_map = {
        "A": "application", "A1": "application", "A2": "application", "A3": "application",
        "A4": "application", "A8": "application", "A9": "application",
        "B": "grant", "B1": "grant", "B2": "grant", "B3": "grant", "B8": "grant", "B9": "grant",
        "C1": "grant", "C2": "grant",
        "U": "utility_model", "U1": "utility_model",
        "S": "design", "S1": "design",
        "Y1": "grant", "Y2": "grant",
        "P": "plant", "P1": "plant", "P2": "plant", "P3": "plant", "P4": "plant",
    }

    kind_clean = df["kind"].astype("string").str.strip().str.upper()
    df["kind_category"] = kind_clean.map(kind_map)

    # Fallback: use first letter of kind code if exact match not found (e.g. unseen "A5" -> "application")
    unmatched = df["kind_category"].isna() & kind_clean.notna()
    first_letter_map = {"A": "application", "B": "grant", "C": "grant",
                         "U": "utility_model", "S": "design", "Y": "grant", "P": "plant"}
    df.loc[unmatched, "kind_category"] = (
        kind_clean.loc[unmatched].str[0].map(first_letter_map)
    )
    df["kind_category"] = df["kind_category"].fillna("other")

    # Family-level aggregation (guard against missing/blank family_id values)
    df["family_id"] = df["family_id"].fillna("UNKNOWN")
    is_unknown_family = df["family_id"] == "UNKNOWN"

    family_counts = df.groupby("family_id")["doc_id"].transform("count")
    df["family_size"] = np.where(is_unknown_family, 1, family_counts)

    df["is_first_in_family"] = (
        df.groupby("family_id")["publication_date"].rank(method="first", ascending=True) == 1
    )
    df.loc[is_unknown_family, "is_first_in_family"] = True

    return df