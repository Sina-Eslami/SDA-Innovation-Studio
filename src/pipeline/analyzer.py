import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

CLEAN_DIR = os.path.join("data", "clean")

_flan_model = None
_flan_tokenizer = None


def _load_flan_model():
    global _flan_model, _flan_tokenizer
    if _flan_model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return _flan_model, _flan_tokenizer


def ask_flan(text: str, prompt: str, max_new_tokens: int = 80) -> str:
    """
    Runs a prompt + context text through FLAN-T5 Small and returns the generated answer.
    """
    model, tokenizer = _load_flan_model()
    full_input = f"{prompt}\n\n{text}"
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _load_clean_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(CLEAN_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: clean file not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _keyword_match(series: pd.Series, keywords: list) -> pd.Series:
    if not keywords:
        return pd.Series([True] * len(series), index=series.index)
    pattern = "|".join([str(k).strip().lower() for k in keywords if str(k).strip()])
    if not pattern:
        return pd.Series([True] * len(series), index=series.index)
    return series.astype("string").str.lower().str.contains(pattern, na=False, regex=True)


def _years_back_mask(dates, years_back):
    dates = pd.to_datetime(dates, errors="coerce", utc=True)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years_back)

    return dates >= cutoff


def filter_catalog(keywords: list) -> pd.DataFrame:
    df = _load_clean_csv("catalog_clean.csv")
    if df.empty or "Appliance" not in df.columns:
        return df
    return df[_keyword_match(df["Appliance"], keywords)].reset_index(drop=True)


def filter_news(keywords: list, years_back: int = None) -> pd.DataFrame:
    df = _load_clean_csv("news_clean.csv")
    if df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    if "keyword" in df.columns:
        mask &= _keyword_match(df["keyword"], keywords)
    if "published_at" in df.columns:
        mask &= _years_back_mask(df["published_at"], years_back)
    return df[mask].reset_index(drop=True)


def filter_patents(keywords: list, years_back: int = None, country=None) -> pd.DataFrame:
    df = _load_clean_csv("patents_clean.csv")
    if df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    if "keyword" in df.columns:
        mask &= _keyword_match(df["keyword"], keywords)
    if "publication_date" in df.columns:
        mask &= _years_back_mask(df["publication_date"], years_back)
    if country and "country_name" in df.columns:
        countries = [country] if isinstance(country, str) else country
        countries_lower = [c.strip().lower() for c in countries]
        mask &= df["country_name"].astype("string").str.lower().isin(countries_lower)
    return df[mask].reset_index(drop=True)


def filter_reviews(keywords: list) -> pd.DataFrame:
    df = _load_clean_csv("reviews_clean.csv")
    if df.empty or "PRODUCT_CATEGORY" not in df.columns:
        return df
    return df[_keyword_match(df["PRODUCT_CATEGORY"], keywords)].reset_index(drop=True)


def filter_social(keywords: list) -> pd.DataFrame:
    df = _load_clean_csv("social_clean.csv")
    if df.empty or "TOPIC" not in df.columns:
        return df
    return df[_keyword_match(df["TOPIC"], keywords)].reset_index(drop=True)


def run_filters(keywords: list, years_back: int = None, country=None) -> dict:
    return {
        "catalog": filter_catalog(keywords),
        "news": filter_news(keywords, years_back),
        "patents": filter_patents(keywords, years_back, country),
        "reviews": filter_reviews(keywords),
        "social": filter_social(keywords),
    }


# --- Summary functions ---

PATENTS_SUMMARY_PROMPT = "Summarize who is filing these patents, the approximate filing trend over time, and the key technical concepts mentioned:"

def summarize_patents(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"text_summary": "No patent data available for this filter.", "volume_by_year": {}, "top_keywords": []}

    volume_by_year = df["pub_year"].value_counts().sort_index().to_dict() if "pub_year" in df.columns else {}
    top_countries = df["country_name"].value_counts().head(5).to_dict() if "country_name" in df.columns else {}

    sample_text = " ".join(df["full_text"].dropna().head(20).tolist()) if "full_text" in df.columns else ""
    nlp_summary = ask_flan(sample_text, PATENTS_SUMMARY_PROMPT) if sample_text else "No text available for NLP summary."

    return {
        "text_summary": nlp_summary,
        "volume_by_year": volume_by_year,
        "top_filing_countries": top_countries,
        "n_patents": len(df),
    }


NEWS_SUMMARY_PROMPT = "Summarize which companies and products are mentioned, and what issues or features are discussed in this news coverage:"

def summarize_news(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"text_summary": "No news data available for this filter.", "n_articles": 0}

    sample_text = " ".join(df["full_text"].dropna().head(20).tolist()) if "full_text" in df.columns else ""
    nlp_summary = ask_flan(sample_text, NEWS_SUMMARY_PROMPT) if sample_text else "No text available for NLP summary."

    avg_sentiment = df["sentiment_polarity"].mean() if "sentiment_polarity" in df.columns else None

    return {
        "text_summary": nlp_summary,
        "n_articles": len(df),
        "avg_sentiment": avg_sentiment,
    }


SOCIAL_SUMMARY_PROMPT = "Summarize the main user pain points, desired features, and overall sentiment discussed in this content:"

def summarize_reviews_social(reviews_df: pd.DataFrame, social_df: pd.DataFrame) -> dict:
    result = {"n_reviews": len(reviews_df), "n_social_posts": len(social_df)}

    result["avg_review_sentiment"] = (
        reviews_df["SENTIMENT_POLARITY"].mean() if not reviews_df.empty and "SENTIMENT_POLARITY" in reviews_df.columns else None
    )
    result["avg_rating"] = (
        reviews_df["RATING"].mean() if not reviews_df.empty and "RATING" in reviews_df.columns else None
    )

    combined_text_parts = []
    if not reviews_df.empty and "FULL_REVIEW_TEXT" in reviews_df.columns:
        combined_text_parts += reviews_df["FULL_REVIEW_TEXT"].dropna().head(15).tolist()
    if not social_df.empty and "FULL_TEXT" in social_df.columns:
        combined_text_parts += social_df["FULL_TEXT"].dropna().head(15).tolist()

    sample_text = " ".join(combined_text_parts)
    result["text_summary"] = ask_flan(sample_text, SOCIAL_SUMMARY_PROMPT) if sample_text else "No text available for NLP summary."

    return result


def summarize_catalog(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_products": 0, "appliance_breakdown": {}, "energy_tier_breakdown": {}}

    return {
        "n_products": len(df),
        "appliance_breakdown": df["Appliance"].value_counts().to_dict() if "Appliance" in df.columns else {},
        "energy_tier_breakdown": df["ENERGY_TIER"].value_counts().to_dict() if "ENERGY_TIER" in df.columns else {},
        "top_brands": df["Brand"].value_counts().head(10).to_dict() if "Brand" in df.columns else {},
    }


# --- D-V-F scoring (placeholder) ---

def calculate_dvf_score(patents_summary: dict, news_summary: dict, reviews_social_summary: dict, catalog_summary: dict) -> dict:
    """
    Placeholder D-V-F scoring. Replace scoring logic once methodology is finalized.
    """
    desirability = 50   # placeholder - e.g. derive from review/social sentiment + pain point frequency
    viability = 50       # placeholder - e.g. derive from catalog competition + news company activity
    feasibility = 50     # placeholder - e.g. derive from patent volume/maturity + technical barriers

    def score_to_label(score):
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    return {
        "desirability_score": desirability,
        "viability_score": viability,
        "feasibility_score": feasibility,
        "desirability_label": score_to_label(desirability),
        "viability_label": score_to_label(viability),
        "feasibility_label": score_to_label(feasibility),
    }


if __name__ == "__main__":
    keywords = ["espresso", "coffee"]
    filtered = run_filters(keywords, years_back=3, country="Germany")

    patents_summary = summarize_patents(filtered["patents"])
    news_summary = summarize_news(filtered["news"])
    reviews_social_summary = summarize_reviews_social(filtered["reviews"], filtered["social"])
    catalog_summary = summarize_catalog(filtered["catalog"])

    dvf = calculate_dvf_score(patents_summary, news_summary, reviews_social_summary, catalog_summary)