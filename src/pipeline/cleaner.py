import os
import pandas as pd
from datetime import datetime

from src.data_builder.patents_clean_fe import clean_patents_df, engineer_patents_features
from src.data_builder.news_clean_fe import clean_news_df, engineer_news_text_features
from src.data_builder.reviews_clean_fe import clean_reviews_df, engineer_reviews_features
from src.data_builder.catalog_clean_fe import clean_catalog_df, engineer_catalog_features
from src.data_builder.social_clean_fe import clean_topic_df, engineer_topic_features


DATA_DIR = "data"
STATIC_DIR = os.path.join(DATA_DIR, "static")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")


def _make_run_folder(run_id: str) -> str:
    folder = os.path.join(CLEAN_DIR, run_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def _save_csv(df: pd.DataFrame, filename: str, folder: str = CLEAN_DIR) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    return path


def clean_dynamic_sources(run_id: str) -> dict:
    """
    Cleans patents + news for a given run_id.
    Reads raw CSVs from data/raw/<run_id>/, writes cleaned+FE CSVs to data/clean/<run_id>/.
    """
    # folder = _make_run_folder(run_id)
    outputs = {}

    patents_raw_path = os.path.join(RAW_DIR, "raw-patent.csv")
    if os.path.exists(patents_raw_path):
        patents_raw = pd.read_csv(patents_raw_path)
        patents_clean = clean_patents_df(patents_raw)
        patents_fe = engineer_patents_features(patents_clean)
        outputs["patents"] = _save_csv(patents_fe, "patents_clean.csv")
    else:
        print(f"Warning: no raw patents file found at {patents_raw_path}")

    news_raw_path = os.path.join(RAW_DIR, "raw-news.csv")
    if os.path.exists(news_raw_path):
        news_raw = pd.read_csv(news_raw_path)
        news_clean = clean_news_df(news_raw)
        news_fe = engineer_news_text_features(news_clean)
        outputs["news"] = _save_csv(news_fe, "news_clean.csv")
    else:
        print(f"Warning: no raw news file found at {news_raw_path}")

    return outputs


def clean_static_sources(run_id: str) -> dict:
    """
    Cleans catalog + reviews + social (static, re-cleaned each run for consistency).
    Reads raw CSVs from data/static/, writes cleaned+FE CSVs to data/clean/<run_id>/.
    """
    # folder = _make_run_folder(run_id)
    outputs = {}

    static_sources = {
        "catalog": ("raw - catalogs.csv", clean_catalog_df, engineer_catalog_features, "catalog_clean.csv"),
        "reviews": ("raw - reviews.csv", clean_reviews_df, engineer_reviews_features, "reviews_clean.csv"),
        "social": ("raw - socials.csv", clean_topic_df, engineer_topic_features, "social_clean.csv"),
    }

    for key, (raw_filename, clean_fn, fe_fn, out_filename) in static_sources.items():
        raw_path = os.path.join(STATIC_DIR, raw_filename)
        if os.path.exists(raw_path):
            raw_df = pd.read_csv(raw_path)
            cleaned = clean_fn(raw_df)
            fe = fe_fn(cleaned)
            outputs[key] = _save_csv(fe, out_filename)
        else:
            print(f"Warning: no static raw file found at {raw_path}")

    return outputs


def run_full_cleaning(run_id: str | None = None) -> dict:
    """
    Main entry point: cleans all dynamic (patents, news) and static (catalog, reviews, social)
    sources for a given run_id, saving all outputs into data/clean/<run_id>/.
    """
    run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results = {}
    results.update(clean_dynamic_sources(run_id))
    results.update(clean_static_sources(run_id))

    print(f"Cleaning complete for run_id={run_id}")
    for source, path in results.items():
        print(f"  {source}: {path}")

    return results


if __name__ == "__main__":
    run_full_cleaning()