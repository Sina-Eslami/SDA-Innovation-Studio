import pandas as pd
import numpy as np
from pathlib import Path

import re

BASE_DIR = Path(__file__).resolve().parent
file_dir = BASE_DIR / ".." / ".." / "data" / "static"

file_path = file_dir / "raw - reviews.csv"

reviews_df = pd.read_csv(file_path)

#-------------cleaning-----------------------------------

def clean_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    for col in ["DOC_ID", "PRODUCT_ID", "PRODUCT_CATEGORY", "LABEL", "VERIFIED_PURCHASE"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})

    for col in ["PRODUCT_TITLE", "REVIEW_TITLE", "REVIEW_TEXT"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
            df[col] = df[col].str.replace(r"[\r\n\t]+", " ", regex=True)
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce")
    df.loc[~df["RATING"].between(1, 5), "RATING"] = np.nan

    def normalize_bool(val):
        if pd.isna(val):
            return np.nan
        v = str(val).strip().lower()
        if v in {"y", "yes", "true", "1", "t"}:
            return True
        if v in {"n", "no", "false", "0", "f"}:
            return False
        return np.nan

    if "VERIFIED_PURCHASE" in df.columns:
        df["VERIFIED_PURCHASE"] = df["VERIFIED_PURCHASE"].apply(normalize_bool)

    if "LABEL" in df.columns:
        df["LABEL"] = df["LABEL"].str.upper()

    if "PRODUCT_CATEGORY" in df.columns:
        df["PRODUCT_CATEGORY"] = df["PRODUCT_CATEGORY"].str.upper().str.replace(r"[_\-]+", " ", regex=True)

    df = df[~(df["REVIEW_TITLE"].isna() & df["REVIEW_TEXT"].isna())]

    critical_cols = [c for c in ["DOC_ID", "PRODUCT_ID", "RATING"] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    if "DOC_ID" in df.columns:
        df = df.drop_duplicates(subset=["DOC_ID"], keep="first")

    dedup_cols = [c for c in ["PRODUCT_ID", "REVIEW_TEXT"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="first")

    for col in ["REVIEW_TITLE", "REVIEW_TEXT", "PRODUCT_TITLE"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df = df.reset_index(drop=True)
    return df

#-------------feature engineering------------------------

def engineer_reviews_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def has_repeated_chars(t):
        return bool(re.search(r"(.)\1{2,}", str(t)))

    def combine_texts(row):
        t = str(row["REVIEW_TITLE"]).strip()
        b = str(row["REVIEW_TEXT"]).strip()
        if t and b and t.lower() != b.lower():
            return f"{t}. {b}"
        return t or b

    df["FULL_REVIEW_TEXT"] = df.apply(combine_texts, axis=1)

    df["REVIEW_TITLE_WORD_COUNT"] = df["REVIEW_TITLE"].str.split().apply(len)
    df["REVIEW_TEXT_WORD_COUNT"] = df["REVIEW_TEXT"].str.split().apply(len)
    df["REVIEW_TEXT_CHAR_LEN"] = df["REVIEW_TEXT"].str.len()
    df["FULL_REVIEW_WORD_COUNT"] = df["FULL_REVIEW_TEXT"].str.split().apply(len)

    df["REVIEW_LENGTH_CATEGORY"] = pd.cut(
        df["REVIEW_TEXT_WORD_COUNT"],
        bins=[-1, 0, 10, 50, np.inf],
        labels=["empty", "short", "medium", "long"]
    )

    df["REVIEW_EXCLAIM_COUNT"] = df["REVIEW_TEXT"].str.count("!")
    df["REVIEW_QUESTION_COUNT"] = df["REVIEW_TEXT"].str.count(r"\?")
    df["REVIEW_CAPS_RATIO"] = df["REVIEW_TEXT"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    df["TITLE_CAPS_RATIO"] = df["REVIEW_TITLE"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    df["HAS_REPEATED_CHARS"] = df["REVIEW_TEXT"].apply(has_repeated_chars)

    df["RATING_CATEGORY"] = pd.cut(
        df["RATING"], bins=[0, 2, 3, 5], labels=["negative", "neutral", "positive"], include_lowest=True
    )
    df["IS_EXTREME_RATING"] = df["RATING"].isin([1, 5])

    product_avg_rating = df.groupby("PRODUCT_ID")["RATING"].transform("mean")
    df["RATING_DEVIATION_FROM_PRODUCT_AVG"] = df["RATING"] - product_avg_rating

    df["TITLE_TEXT_LEN_RATIO"] = df["REVIEW_TITLE_WORD_COUNT"] / df["REVIEW_TEXT_WORD_COUNT"].replace(0, np.nan)

    df["UNVERIFIED_EXTREME"] = (~df["VERIFIED_PURCHASE"].fillna(False).infer_objects(copy=False)) & df["IS_EXTREME_RATING"]

    df["HAS_SUBSTANTIAL_TEXT"] = df["REVIEW_TEXT_WORD_COUNT"] >= 5

    df["PRODUCT_REVIEW_COUNT"] = df.groupby("PRODUCT_ID")["DOC_ID"].transform("count")

    return df