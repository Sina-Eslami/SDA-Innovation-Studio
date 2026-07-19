import os
import re
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords

import spacy

CLEAN_DIR = os.path.join("data", "clean")

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    STOPWORDS = set(stopwords.words("english"))

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm")


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


def _years_back_mask(series: pd.Series, years_back) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", utc=True)
    if years_back is None:
        return pd.Series([True] * len(series), index=series.index)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years_back)
    return dates >= cutoff


# ---------- Filters (unchanged) ----------

def filter_catalog(keywords: list) -> pd.DataFrame:
    df = _load_clean_csv("catalog_clean.csv")
    if df.empty or "Appliance" not in df.columns:
        return df
    return df[_keyword_match(df["Appliance"], keywords)].reset_index(drop=True)


def filter_news(keywords: list, years_back=None) -> pd.DataFrame:
    df = _load_clean_csv("news_clean.csv")
    if df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    if "keyword" in df.columns:
        mask &= _keyword_match(df["keyword"], keywords)
    if "published_at" in df.columns:
        mask &= _years_back_mask(df["published_at"], years_back)
    return df[mask].reset_index(drop=True)


def filter_patents(keywords: list, years_back=None, country=None) -> pd.DataFrame:
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


def run_filters(keywords: list, years_back=None, country=None) -> dict:
    return {
        "catalog": filter_catalog(keywords),
        "news": filter_news(keywords, years_back),
        "patents": filter_patents(keywords, years_back, country),
        "reviews": filter_reviews(keywords),
        "social": filter_social(keywords),
    }


# ---------- NLP extraction helpers (sklearn + spaCy + NLTK) ----------

def extract_top_tfidf_terms(texts: list, top_n: int = 15, ngram_range=(1, 2)) -> list:
    """Key concepts/keywords via TF-IDF ranking (used for patents)."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, max_features=2000, min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [term for term, _ in ranked[:top_n]]


def extract_noun_phrases(texts: list, top_n: int = 15) -> list:
    """Discussed aspects/issues/features via spaCy noun chunks (used for news, reviews, social)."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    counter = Counter()
    for doc in _nlp.pipe(texts[:200], disable=["ner"]):
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            phrase = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", phrase)
            if len(phrase) > 2 and phrase not in STOPWORDS:
                counter[phrase] += 1
    return [phrase for phrase, _ in counter.most_common(top_n)]


PAIN_POINT_PATTERNS = [
    "doesn't work", "does not work", "stopped working", "broke", "broken",
    "too loud", "too noisy", "hard to clean", "difficult to clean", "leaks",
    "leaking", "slow", "expensive", "poor quality", "waste of money",
    "disappointed", "not worth", "malfunction", "defective", "stopped",
]

DESIRE_PATTERNS = [
    "wish it had", "should have", "would be nice", "hope they add",
    "needs a", "needs to have", "want a", "would love", "should include",
    "missing a", "if only it had",
]


def extract_pain_points_and_desires(texts: list, top_n: int = 10) -> dict:
    texts = [str(t).lower() for t in texts if isinstance(t, str) and t.strip()]
    pain_phrase_counter = Counter()
    desire_phrase_counter = Counter()

    pain_volume = 0
    desire_volume = 0

    for text in texts:
        for pattern in PAIN_POINT_PATTERNS:
            for match in re.finditer(re.escape(pattern), text):
                start = match.end()
                snippet = text[start:start + 40].strip()
                snippet = re.split(r"[.!?,;]", snippet)[0].strip()
                if snippet:
                    pain_phrase_counter[f"{pattern} -> {snippet}"] += 1
                    pain_volume += len(snippet)

        for pattern in DESIRE_PATTERNS:
            for match in re.finditer(re.escape(pattern), text):
                start = match.end()
                snippet = text[start:start + 40].strip()
                snippet = re.split(r"[.!?,;]", snippet)[0].strip()
                if snippet:
                    desire_phrase_counter[f"{pattern} -> {snippet}"] += 1
                    desire_volume += len(snippet)

    noun_phrases = extract_noun_phrases(texts, top_n=top_n)

    desirability_score = (
        round(desire_volume / pain_volume, 3) if pain_volume > 0 else None
    )

    return {
        "top_pain_signals": [p for p, _ in pain_phrase_counter.most_common(top_n)],
        "top_desire_signals": [p for p, _ in desire_phrase_counter.most_common(top_n)],
        "top_discussed_aspects": noun_phrases,
        "pain_volume": pain_volume,
        "desire_volume": desire_volume,
        "desirability_score": desirability_score,
    }


def cluster_texts_kmeans(texts: list, n_clusters: int = 3) -> dict:
    """Optional: groups texts into themes via TF-IDF + KMeans, returns top terms per cluster."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < n_clusters:
        return {}
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(matrix)
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for i in range(n_clusters):
        center = km.cluster_centers_[i]
        top_idx = center.argsort()[-8:][::-1]
        cluster_terms[f"cluster_{i}"] = [terms[j] for j in top_idx]
    return cluster_terms


# ---------- Summary functions ----------

def summarize_patents(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_patents": 0, "volume_by_year": {}, "top_filing_countries": {}, "top_applicants": {}, "key_concepts_tfidf": [], "key_concepts_noun_phrases": []}

    volume_by_year = df["pub_year"].value_counts().sort_index().to_dict() if "pub_year" in df.columns else {}
    top_countries = df["country_name"].value_counts().head(5).to_dict() if "country_name" in df.columns else {}
    top_applicants = df["family_id"].value_counts().head(10).to_dict() if "family_id" in df.columns else {}

    texts = df["full_text"].dropna().tolist() if "full_text" in df.columns else []

    return {
        "n_patents": len(df),
        "volume_by_year": volume_by_year,
        "top_filing_countries": top_countries,
        "top_applicants": top_applicants,
        "key_concepts_tfidf": extract_top_tfidf_terms(texts, top_n=15),
        "key_concepts_noun_phrases": extract_noun_phrases(texts, top_n=15),
    }


def summarize_news(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_articles": 0, "companies_mentioned": [], "issues_and_features": []}

    texts = df["full_text"].dropna().tolist() if "full_text" in df.columns else []

    companies_mentioned = []
    if texts:
        org_counter = Counter()
        for doc in _nlp.pipe(texts[:200]):
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org_counter[ent.text.strip()] += 1
        companies_mentioned = [org for org, _ in org_counter.most_common(15)]

    return {
        "n_articles": len(df),
        "companies_mentioned": companies_mentioned,
        "issues_and_features": extract_noun_phrases(texts, top_n=15),
        "avg_sentiment": df["sentiment_polarity"].mean() if "sentiment_polarity" in df.columns else None,
    }


def summarize_reviews_social(reviews_df: pd.DataFrame, social_df: pd.DataFrame) -> dict:
    result = {"n_reviews": len(reviews_df), "n_social_posts": len(social_df)}

    result["avg_review_sentiment"] = (
        reviews_df["SENTIMENT_POLARITY"].mean() if not reviews_df.empty and "SENTIMENT_POLARITY" in reviews_df.columns else None
    )
    result["avg_social_sentiment"] = (
        social_df["SENTIMENT_POLARITY"].mean() if not social_df.empty and "SENTIMENT_POLARITY" in social_df.columns else None
    )
    result["avg_rating"] = (
        reviews_df["RATING"].mean() if not reviews_df.empty and "RATING" in reviews_df.columns else None
    )

    combined_texts = []
    if not reviews_df.empty and "FULL_REVIEW_TEXT" in reviews_df.columns:
        combined_texts += reviews_df["FULL_REVIEW_TEXT"].dropna().tolist()
    if not social_df.empty and "FULL_TEXT" in social_df.columns:
        combined_texts += social_df["FULL_TEXT"].dropna().tolist()

    pain_desire = extract_pain_points_and_desires(combined_texts, top_n=10)
    result.update(pain_desire)
    result["desirability"] = (
    100 if pain_desire["pain_volume"] == 0
    else 100*(pain_desire["desire_volume"] / pain_desire["pain_volume"])
    )

    return result


def summarize_catalog(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_products": 0, "appliance_breakdown": {}, "energy_tier_breakdown": {}, "top_brands": {}}

    return {
        "n_products": len(df),
        "appliance_breakdown": df["Appliance"].value_counts().to_dict() if "Appliance" in df.columns else {},
        "energy_tier_breakdown": df["ENERGY_TIER"].value_counts().to_dict() if "ENERGY_TIER" in df.columns else {},
        "top_brands": df["Brand"].value_counts().head(6).to_dict() if "Brand" in df.columns else {},
    }


# ---------- D-V-F scoring (placeholder) ----------

def calculate_dvf_score(patents_summary: dict, news_summary: dict, reviews_social_summary: dict, catalog_summary: dict) -> dict:
    desirability = reviews_social_summary.get("desirability")
    viability = (reviews_social_summary.get("desirability")/3) + 100*(reviews_social_summary.get("avg_social_sentiment")/3) + ((reviews_social_summary.get("avg_rating")-1)*25/3)
    feasibility = 61

    def score_to_label(score):
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    return {
        "desirability_score": desirability, "viability_score": viability, "feasibility_score": feasibility,
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