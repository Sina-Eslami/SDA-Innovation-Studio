import streamlit.components.v1 as components
import json
from pathlib import Path


def innovation_spider(
    desirability,
    viability,
    feasibility,
    social_sentiment,
    reviews_rating,
):

    html_path = Path(__file__).parent / "spider_chart.html"

    html = html_path.read_text()

    data = {
        "desirability": desirability,
        "viability": viability,
        "feasibility": feasibility,
        "social_sentiment": social_sentiment,
        "reviews_rating": reviews_rating,
    }


    html = html.replace(
        "__VALUES__",
        json.dumps(data)
    )


    components.html(
        html,
        height=500,
        scrolling=False
    )