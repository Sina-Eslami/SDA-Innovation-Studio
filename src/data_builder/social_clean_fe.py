import pandas as pd
import numpy as np

from pathlib import Path
from textblob import TextBlob


BASE_DIR = Path(__file__).resolve().parent
file_dir = BASE_DIR / ".." / ".." / "data" / "static"

social_df = pd.read_csv(file_dir / 'raw - socials.csv')

def clean_topic_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["absolute", "block"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
        df[col] = df[col].replace({"": np.nan, "None": np.nan, "nan": np.nan})

    df["TITLE"] = df["absolute"].fillna(df["block"])

    mismatch = (
        df["absolute"].notna() & df["block"].notna() &
        (df["absolute"].str.lower() != df["block"].str.lower())
    )
    df.loc[mismatch, "TITLE"] = df.loc[mismatch, "absolute"]

    df = df.drop(columns=["absolute", "block"])

    df["md"] = df["md"].astype("string").str.strip()
    df["md"] = df["md"].str.replace(r"\s+", " ", regex=True)
    df["md"] = df["md"].replace({"": np.nan, "None": np.nan, "nan": np.nan})
    df = df.rename(columns={"md": "BODY_TEXT"})

    df["topic"] = df["topic"].astype("string").str.strip().str.upper()
    df = df.rename(columns={"topic": "TOPIC"})

    df = df.dropna(subset=["TITLE", "BODY_TEXT", "TOPIC"])

    df["_title_norm"] = df["TITLE"].str.lower()
    df = df.drop_duplicates(subset=["_title_norm", "TOPIC"], keep="first")
    df = df.drop(columns=["_title_norm"])

    return df.reset_index(drop=True)


def engineer_topic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TITLE_WORD_COUNT"] = df["TITLE"].str.split().apply(len)
    df["BODY_WORD_COUNT"] = df["BODY_TEXT"].str.split().apply(len)
    df["BODY_CHAR_LEN"] = df["BODY_TEXT"].str.len()

    df["FULL_TEXT"] = (df["TITLE"] + ". " + df["BODY_TEXT"]).str.strip()

    df["BODY_LENGTH_CATEGORY"] = pd.cut(
        df["BODY_WORD_COUNT"], bins=[-1, 20, 100, 300, np.inf],
        labels=["very_short", "short", "medium", "long"]
    )

    df["TITLE_EXCLAIM_COUNT"] = df["TITLE"].str.count("!")
    df["TITLE_CAPS_RATIO"] = df["TITLE"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    df["TOPIC_ARTICLE_COUNT"] = df.groupby("TOPIC")["TITLE"].transform("count")
    df["TOPIC_AVG_BODY_LEN"] = df.groupby("TOPIC")["BODY_WORD_COUNT"].transform("mean")
    df["BODY_LEN_VS_TOPIC_AVG"] = df["BODY_WORD_COUNT"] - df["TOPIC_AVG_BODY_LEN"]

    # sentiment

    sentiment_scores = df["FULL_TEXT"].apply(
    lambda t: TextBlob(str(t)).sentiment if str(t).strip() else (0.0, 0.0))
    df["SENTIMENT_POLARITY"] = sentiment_scores.apply(lambda s: s[0])
    df["SENTIMENT_SUBJECTIVITY"] = sentiment_scores.apply(lambda s: s[1])
    df["SENTIMENT_LABEL"] = pd.cut(
        df["SENTIMENT_POLARITY"], bins=[-1.01, -0.1, 0.1, 1.01],
        labels=["negative", "neutral", "positive"]
    )

    return df