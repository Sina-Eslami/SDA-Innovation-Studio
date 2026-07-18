import streamlit as st

from src.prompt_reader import parse_prompt
from src.pipeline.analyzer import (
    run_filters,
    summarize_patents,
    summarize_news,
    summarize_reviews_social,
    summarize_catalog,
)

st.set_page_config(page_title="SDA Innovation Studio", layout="wide")

CUSTOM_CSS = """
<style>
.main { background-color: #f7f8fa; }

.app-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1a1a2e;
    margin-bottom: 0.2rem;
}

.app-subtitle {
    font-size: 1.05rem;
    color: #52555a;
    max-width: 900px;
    line-height: 1.5;
    margin-bottom: 1.8rem;
}

.control-row {
    background-color: #ffffff;
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    margin-bottom: 2rem;
}

.summary-card {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 1.4rem 1.3rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    min-height: 340px;
    margin-bottom: 1.5rem;
    border: 1px solid #eef0f3;
}

.summary-card h4 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #2b2d42;
    margin-bottom: 0.8rem;
    border-bottom: 2px solid #f0f0f5;
    padding-bottom: 0.5rem;
}

.signal-box {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    min-height: 220px;
    border: 1px solid #eef0f3;
}

.signal-box h4 {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2b2d42;
    margin-bottom: 0.9rem;
}

.tag {
    display: inline-block;
    background-color: #eef2ff;
    color: #3730a3;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.25rem 0.2rem 0;
}

.tag-pain { background-color: #fee2e2; color: #991b1b; }
.tag-desire { background-color: #dcfce7; color: #166534; }

.metric-line { font-size: 0.85rem; color: #555; margin-bottom: 0.4rem; }

.stButton>button {
    border-radius: 12px;
    background-color: #4f46e5;
    color: white;
    font-weight: 600;
    padding: 0.55rem 1.6rem;
    border: none;
}

.stButton>button:hover { background-color: #4338ca; color: white; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="app-title">SDA Innovation Studio</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-subtitle">
    This app performs multi-source NLP analysis across patents, news, product reviews, social media,
    and product catalogs to surface innovation-relevant signals. Enter a theme below to gather intelligence
    on market demand and existing supply, and get a lightweight Desirability, Viability, and Feasibility (D-V-F)
    read on the opportunity space.
    </div>
    """,
    unsafe_allow_html=True,
)

COUNTRY_OPTIONS = [
    "All countries", "Germany", "United States",
    "World Intellectual Property Organization (PCT)", "European Patent Office",
    "Japan", "United Kingdom", "China", "South Korea", "Turkey", "Sweden",
    "Taiwan", "Australia", "Moldova", "Italy",
]

col1, col2, col3, col4 = st.columns([3, 1.3, 1.6, 1])

with col1:
    theme_input = st.text_input("Theme", placeholder="e.g. espresso machine, robot vacuum, air fryer...", label_visibility="collapsed")

with col2:
    years_back_label = st.selectbox("Time window", ["1 year back", "2 years back", "3 years back"], label_visibility="collapsed")

with col3:
    country_choice = st.selectbox("Country", COUNTRY_OPTIONS, label_visibility="collapsed")

with col4:
    gather_clicked = st.button("Gather Intel", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

years_back_map = {"1 year back": 1, "2 years back": 2, "3 years back": 3}
years_back = years_back_map[years_back_label]
country = None if country_choice == "All countries" else country_choice


def render_dict_list(items, css_class="tag"):
    if not items:
        return "<span class='metric-line'>No data available.</span>"
    return "".join([f"<span class='{css_class}'>{str(i)}</span>" for i in items])


def render_metric_lines(pairs):
    if not pairs:
        return "<div class='metric-line'>No data available.</div>"
    lines = ""
    for k, v in pairs:
        lines += f"<div class='metric-line'><b>{k}</b>: {v}</div>"
    return lines


if gather_clicked:
    keywords = parse_prompt(theme_input)

    with st.spinner("Gathering intel across sources..."):
        filtered = run_filters(keywords, years_back=years_back, country=country)
        patents_summary = summarize_patents(filtered["patents"])
        news_summary = summarize_news(filtered["news"])
        reviews_social_summary = summarize_reviews_social(filtered["reviews"], filtered["social"])
        catalog_summary = summarize_catalog(filtered["catalog"])

    st.session_state["patents_summary"] = patents_summary
    st.session_state["news_summary"] = news_summary
    st.session_state["reviews_social_summary"] = reviews_social_summary
    st.session_state["catalog_summary"] = catalog_summary
    st.session_state["has_results"] = True


if st.session_state.get("has_results"):
    patents_summary = st.session_state["patents_summary"]
    news_summary = st.session_state["news_summary"]
    reviews_social_summary = st.session_state["reviews_social_summary"]
    catalog_summary = st.session_state["catalog_summary"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        top_countries_str = ", ".join(list(patents_summary.get("top_filing_countries", {}).keys())[:3]) or "N/A"
        card_html = f"""
        <div class="summary-card">
            <h4>Patent Summary</h4>
            <div class="metric-line"><b>Patents found</b>: {patents_summary.get("n_patents", 0)}</div>
            <div class="metric-line"><b>Top filing countries</b>: {top_countries_str}</div>
            <div class="metric-line"><b>Key concepts</b></div>
            {render_dict_list(patents_summary.get("key_concepts_tfidf", [])[:8])}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    with c2:
        avg_sent = news_summary.get("avg_sentiment")
        avg_sent_str = f"{avg_sent:.2f}" if avg_sent is not None else "N/A"
        card_html = f"""
        <div class="summary-card">
            <h4>News Summary</h4>
            <div class="metric-line"><b>Articles found</b>: {news_summary.get("n_articles", 0)}</div>
            <div class="metric-line"><b>Avg sentiment</b>: {avg_sent_str}</div>
            <div class="metric-line"><b>Companies mentioned</b></div>
            {render_dict_list(news_summary.get("companies_mentioned", [])[:8])}
            <div class="metric-line"><b>Issues & features</b></div>
            {render_dict_list(news_summary.get("issues_and_features", [])[:8])}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    with c3:
        avg_r_sent = reviews_social_summary.get("avg_review_sentiment")
        avg_rating = reviews_social_summary.get("avg_rating")
        avg_r_sent_str = f"{avg_r_sent:.2f}" if avg_r_sent is not None else "N/A"
        avg_rating_str = f"{avg_rating:.2f}" if avg_rating is not None else "N/A"
        card_html = f"""
        <div class="summary-card">
            <h4>Reviews & Social Summary</h4>
            <div class="metric-line"><b>Reviews found</b>: {reviews_social_summary.get("n_reviews", 0)}</div>
            <div class="metric-line"><b>Social posts found</b>: {reviews_social_summary.get("n_social_posts", 0)}</div>
            <div class="metric-line"><b>Avg rating</b>: {avg_rating_str}</div>
            <div class="metric-line"><b>Avg review sentiment</b>: {avg_r_sent_str}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    with c4:
        card_html = f"""
        <div class="summary-card">
            <h4>Catalogs Summary</h4>
            <div class="metric-line"><b>Products found</b>: {catalog_summary.get("n_products", 0)}</div>
            <div class="metric-line"><b>Appliance types</b></div>
            {render_dict_list(list(catalog_summary.get("appliance_breakdown", {}).keys())[:8])}
            <div class="metric-line"><b>Top brands</b></div>
            {render_dict_list(list(catalog_summary.get("top_brands", {}).keys())[:8])}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        signal_html = f"""
        <div class="signal-box">
            <h4>Demand Signals</h4>
            <div class="metric-line"><b>Top pain points</b></div>
            {render_dict_list(reviews_social_summary.get("top_pain_signals", []), css_class="tag tag-pain")}
            <div class="metric-line" style="margin-top:0.8rem;"><b>Desired features</b></div>
            {render_dict_list(reviews_social_summary.get("top_desire_signals", []), css_class="tag tag-desire")}
            <div class="metric-line" style="margin-top:0.8rem;"><b>Discussed aspects</b></div>
            {render_dict_list(reviews_social_summary.get("top_discussed_aspects", []))}
        </div>
        """
        st.markdown(signal_html, unsafe_allow_html=True)

    with d2:
        signal_html = f"""
        <div class="signal-box">
            <h4>Supply Signals</h4>
            <div class="metric-line"><b>Existing product types</b></div>
            {render_dict_list(list(catalog_summary.get("appliance_breakdown", {}).keys()))}
            <div class="metric-line" style="margin-top:0.8rem;"><b>Key patent concepts</b></div>
            {render_dict_list(patents_summary.get("key_concepts_noun_phrases", [])[:10])}
            <div class="metric-line" style="margin-top:0.8rem;"><b>Companies active in news</b></div>
            {render_dict_list(news_summary.get("companies_mentioned", [])[:10])}
        </div>
        """
        st.markdown(signal_html, unsafe_allow_html=True)