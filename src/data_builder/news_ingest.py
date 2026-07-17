import os
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class NewsClient:
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str | None = None, default_keywords: list[str] | None = None):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not set in environment or passed to NewsClient.")
        self.default_keywords = default_keywords or [
            "air purifier",
            "air fryer",
            "coffee machine",
            "robot vacuum",
        ]

    def _build_params(self, keywords: list[str], from_date: str, to_date: str) -> dict:
        quoted = [f"\"{kw}\"" for kw in keywords]
        query = " OR ".join(quoted)

        return {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "searchIn": "title",
        }

    def _request(self, params: dict) -> dict:
        headers = {"X-Api-Key": self.api_key}
        response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()

    def fetch_last_n_years(self, years_back: int, keywords: list[str] | None = None) -> pd.DataFrame:
        today = date.today()
        from_date = (today - timedelta(days=years_back * 365)).isoformat()
        to_date = today.isoformat()
        kw = keywords or self.default_keywords

        params = self._build_params(kw, from_date, to_date)
        data = self._request(params)

        articles = data.get("articles", [])

        rows = []
        for a in articles:
            title = a.get("title") or ""
            description = a.get("description") or ""
            content = a.get("content") or ""

            combined = " ".join([title, description, content])
            matched = find_matching_keywords(combined, kw)

            rows.append({
                "source_name": a.get("source", {}).get("name"),
                "author": a.get("author"),
                "title": title,
                "description": description,
                "url": a.get("url"),
                "published_at": a.get("publishedAt"),
                "content": content,
                "matched_keywords": ", ".join(matched),   # or keep as list
                "n_matched_keywords": len(matched),
            })

        df = pd.DataFrame(rows)

        output_dir = Path("../../data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / "raw-news.csv", index=False)
        return df

###------Helper Functions-------------------------------------------------

def find_matching_keywords(text: str, keywords: list[str]) -> list[str]:
    if not text:
        return []
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]