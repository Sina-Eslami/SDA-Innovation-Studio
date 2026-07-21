import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pycountry
import math
import plotly.graph_objects as go

import streamlit.components.v1 as components

from src.components.spider_chart import innovation_spider
from src.components.score_summary import score_summary

from src.prompt_reader import parse_prompt
from src.pipeline.nlp_analyzer import (
    run_filters,
    summarize_patents,
    summarize_news,
    summarize_reviews_social,
    summarize_catalog,
    calculate_dvf_score,
)

st.set_page_config(page_title="SDA Innovation Studio", layout="wide")

CUSTOM_CSS = """
<style>
.main { background-color: #f7f8fa; }

.app-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: "#2b4fc9";
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


/* Geographic table styles */
.geo-table {
    padding-top: 0.5rem;
}

.geo-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #9098a8;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #eef0f5;
}

.geo-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid #f5f6f9;
    font-size: 0.85rem;
}

.geo-country {
    flex: 1.4;
    color: #2b2d42;
    font-weight: 500;
}

.geo-count {
    flex: 0.6;
    color: #2b2d42;
    font-weight: 600;
    text-align: right;
}

.geo-bar-track {
    flex: 1;
    height: 4px;
    background-color: #eef0f5;
    border-radius: 4px;
    margin-left: 0.8rem;
    overflow: hidden;
}

.geo-bar-fill {
    height: 100%;
    background-color: #4f46e5;
    border-radius: 4px;
}

.concept-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 12px 4px 4px 4px;
}
.concept-chip {
    background-color: #eef0f5;
    color: #2b4fc9;
    font-size: 12px;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 999px;
    white-space: nowrap;
}

</style>
"""
#-----------Helper functions---------------------
def _country_name_to_iso3(name: str) -> str:
    special_cases = {
        "World Intellectual Property Organization (PCT)": None,
        "European Patent Office": None,
        "South Korea": "KOR",
        "Taiwan": "TWN",
        "Moldova": "MDA",
    }
    if name in special_cases:
        return special_cases[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def build_patents_map_data(patents_df: pd.DataFrame) -> pd.DataFrame:
    if patents_df.empty or "country_name" not in patents_df.columns:
        return pd.DataFrame(columns=["country_name", "iso3", "patent_count"])

    counts = patents_df["country_name"].value_counts().reset_index()
    counts.columns = ["country_name", "patent_count"]
    counts["iso3"] = counts["country_name"].apply(_country_name_to_iso3)
    counts = counts.dropna(subset=["iso3"])
    return counts

def get_top_keywords(df: pd.DataFrame, top_n: int = 12) -> list[str]:
    if "keywords" not in df.columns or df.empty:
        return []
    all_kw = df["keywords"].dropna().str.split(",").explode().str.strip()
    return all_kw.value_counts().head(top_n).index.tolist()

#--------guages functions-------------

def rating_guage(rating):
    """
    rating: float between 1 and 5
    """

    if not isinstance(rating, (int, float)):
        rating = 3

    rating = max(1, min(5, rating))

    radius = 100
    stroke = 35
    circumference = 2 * math.pi * radius

    # Fill the FULL circle clockwise
    progress = (rating - 1) / 4
    filled = progress * circumference

    html = f"""
    <div style="
        width:240;
        height:240;
        margin:auto;
        display:flex;
        justify-content:center;
        align-items:center;
    ">
    <svg width="240" height="240">

        <defs>
        <linearGradient id="ringGradient"
                        x1="0%" y1="0%"
                        x2="100%" y2="100%">
            <stop offset="0%" stop-color="#2b4fc9"/>
            <stop offset="50%" stop-color="#6b8fe0"/>
            <stop offset="100%" stop-color="#e3e8f7"/>
        </linearGradient>
        </defs>

        <circle
            cx="120"
            cy="120"
            r="{radius}"
            fill="none"
            stroke="#E5E7EB"
            stroke-width="{stroke}"
        />

        <!-- Filled ring -->
        <circle
            cx="120"
            cy="120"
            r="{radius}"
            fill="none"
            stroke="url(#ringGradient)"
            stroke-width="{stroke}"
            stroke-linecap="round"
            stroke-dasharray="{filled} {circumference}"
            transform="rotate(-90 120 120)"
        />

        <!-- Center text -->
        <text
            x="120"
            y="85"
            text-anchor="middle"
            font-size="14"
            fill="#666"
            font-family="Arial">
            Rating:
        </text>

        <text
            x="120"
            y="138"
            text-anchor="middle"
            font-size="32"
            font-weight="700"
            fill="#111"
            font-family="Arial">
            {rating:.1f}
        </text>

    </svg>
    </div>
    """

    components.html(html, height=250)

def sentiment_gauge(value):
    """
    value: float between -1 and 1
    """

    value = max(-1, min(1, value))

    radius = 100
    stroke = 35
    circumference = 2 * 3.14159265359 * radius

    filled = abs(value) * circumference / 2

    if value >= 0:
        transform = "rotate(-90 120 120)"
    else:
        transform = "rotate(-90 120 120) scale(-1 1) translate(-240 0)"

    html = f"""
    <div style="
        width:240px;
        height:240px;
        margin:auto;
        display:flex;
        justify-content:center;
        align-items:center;
    ">
    <svg width="240" height="240">

        <circle
            cx="120"
            cy="120"
            r="{radius}"
            fill="none"
            stroke="#E5E7EB"
            stroke-width="{stroke}"
        />

        <!-- Colored arc -->
        <circle
            cx="120"
            cy="120"
            r="{radius}"
            fill="none"
            stroke="#107336"
            stroke-width="{stroke}"
            stroke-linecap="round"
            stroke-dasharray="{filled} {circumference}"
            transform="{transform}"
        />

        <!-- Center text -->
        <text
            x="120"
            y="85"
            text-anchor="middle"
            font-size="14"
            fill="#666"
            font-family="Arial">
            Sentiment:
        </text>

        <text
            x="120"
            y="138"
            text-anchor="middle"
            font-size="32"
            font-weight="700"
            fill="#111"
            font-family="Arial">
            {value:+.2f}
        </text>

    </svg>
    </div>
    """
    components.html(html, height=250)

# ----------catalog plot--------------------

def catalog_bar_plot(data_dict):
    x = list(data_dict.keys())
    y = list(data_dict.values())

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker=dict(
                color=y,
                colorscale=[
                    [0, "#e3e8f7"],
                    [0.5, "#6b8fe0"],
                    [1, "#2b4fc9"],
                ],
                line=dict(
                    color="#2b4fc9",
                    width=2
                ),
            ),

            # value labels
            text=y,
            textposition="outside",
            textfont=dict(
                size=16,
                color="#333"
            )
        )
    )

    fig.update_layout(
        height=400,

        # cartoon bulky feeling
        bargap=0.35,

        plot_bgcolor="white",
        paper_bgcolor="white",

        font=dict(
            family="Arial",
            size=14,
            color="#333"
        ),

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14)
        ),

        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        ),

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=40
        )
    )

    fig.update_traces(
        marker_cornerradius=15
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def catalog_pie_chart(data):

    labels = list(data.keys())
    values = list(data.values())

    colors = {
        "low": "#e3e8f7",
        "medium": "#6b8fe0",
        "high": "#2b4fc9"
    }

    pie_colors = [
        colors.get(label, "#6b8fe0")
        for label in labels
    ]

    total = sum(values)

    text_colors = [
        "white" if label == "high" else "#333"
        for label in labels
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,

                marker=dict(
                    colors=pie_colors,
                    line=dict(
                        color="white",
                        width=5
                    )
                ),

                textinfo="label+percent",

                insidetextfont=dict(
                    size=14,
                    color=text_colors
                ),

                sort=False
            )
        ]
    )


    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:14px'>Total</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            size=32,
            color="#222"
        )
    )


    fig.update_layout(
        height=380,

        showlegend=False,

        paper_bgcolor="white",
        plot_bgcolor="white",

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def render_patents_globe(patents_df: pd.DataFrame):
    map_data = build_patents_map_data(patents_df)

    if map_data.empty:
        st.info("No country-level patent data available for this filter.")
        return

    fig = go.Figure(
        data=go.Choropleth(
            locations=map_data["iso3"],
            z=map_data["patent_count"],
            text=map_data["country_name"],
            colorscale=[[0, "#e3e8f7"], [0.5, "#6b8fe0"], [1, "#2b4fc9"]],
            marker_line_color="white",
            marker_line_width=0.6,
            colorbar_title="Patents",
            hovertemplate="<b>%{text}</b><br>Patents: %{z}<extra></extra>",
        )
    )

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#eef0f5",
        showocean=True,
        oceancolor="#ffffff",
        showcountries=True,
        countrycolor="#d8dce6",
        showframe=False,
        showcoastlines=False,
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )

    col_sources, col_assess = st.columns([2, 1])
    with col_sources:
        st.markdown('<div class="summary-card" style="min-height:auto;"><h4>Patents by Country</h4></div>', unsafe_allow_html=True)
        
        col_map, col_table = st.columns([2.2, 1])

        with col_map:
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displayModeBar": False})

        with col_table:
            table_df = (
                map_data.sort_values("patent_count", ascending=False)
                .head(10)
                [["country_name", "patent_count"]]
                .copy()
            )

            table_df.columns = ["Country", "Patents"]

            st.dataframe(
                table_df,
                column_config={
                    "Country": st.column_config.TextColumn(
                        "Country"
                    ),
                    "Patents": st.column_config.ProgressColumn(
                        "Patents",
                        format="%d",
                        min_value=0,
                        max_value=int(table_df["Patents"].max())
                    )
                },
                hide_index=True,
                width="stretch"
            )

            key_concepts = get_top_keywords(patents_df, top_n=12) 

            chips_html = ""

            for kw in key_concepts:
                clean_kw = str(kw)
                clean_kw = clean_kw.replace("[", "")
                clean_kw = clean_kw.replace("]", "")
                clean_kw = clean_kw.replace("'", "")

                chips_html += f'<span class="concept-chip">{clean_kw}</span>'

            concepts_box_html = f"""
            <div class="geo-table" style="margin-top: 16px;">
                <div class="geo-header">
                    <div>PATENT'S KEY CONCEPTS</div>
                </div>
                <div class="concept-wrap">
                    {chips_html}
                </div>
            </div>
            """
            st.markdown(concepts_box_html, unsafe_allow_html=True)

        col_social , col_review = st.columns([1, 1])
        with col_social:
            st.markdown('<div class="summary-card" style="min-height:auto;"><h4>Social Sentiments</h4></div>', unsafe_allow_html=True)
            sentiment_gauge(reviews_social_summary.get("avg_social_sentiment"))
            signal_html = f"""
            <div class="signal-box">
                <div class="metric-line" style="margin-top:0.8rem;"><b>Desired features</b></div>
                {render_dict_list(reviews_social_summary.get("top_desire_signals", []), css_class="tag tag-desire")}
            </div>
            """
            st.markdown(signal_html, unsafe_allow_html=True)
        with col_review:
            st.markdown('<div class="summary-card" style="min-height:auto;"><h4>Review Ratings</h4></div>', unsafe_allow_html=True)
            rating_guage(reviews_social_summary.get("avg_rating"))
            signal_html = f"""
            <div class="signal-box">
                <div class="metric-line"><b>Top pain points</b></div>
                {render_dict_list(reviews_social_summary.get("top_pain_signals", []), css_class="tag tag-pain")}
            </div>
            """
            st.markdown(signal_html, unsafe_allow_html=True)
        # signal_html = f"""
        # <div class="signal-box">
        #     <div class="metric-line" style="margin-top:0.8rem;"><b>Discussed aspects</b></div>
        #     {render_dict_list(reviews_social_summary.get("top_discussed_aspects", []))}
        # </div>
        # """
        # st.markdown(signal_html, unsafe_allow_html=True)

        st.markdown('<div class="summary-card" style="min-height:auto;"><h4>Catalog</h4></div>', unsafe_allow_html=True)
        col_brands, col_energy = st.columns([1,1])
        with col_brands:
            top_brand_text = f"""
                <div class="geo-header">
                    <div>TOP BRANDS:</div>
                </div>
            """
            st.markdown(top_brand_text, unsafe_allow_html=True)
            catalog_bar_plot(catalog_summary.get("top_brands"))
        with col_energy:
            energy_tiers_text = f"""
                <div class="geo-header">
                    <div>ENERGY TIERS:</div>
                </div>
            """
            st.markdown(energy_tiers_text, unsafe_allow_html=True)
            catalog_pie_chart(catalog_summary.get('energy_tier_breakdown'))

    with col_assess:
        st.markdown('<div class="summary-card" style="min-height:auto;"><h4>Opportunity Assessment</h4></div>', unsafe_allow_html=True)

        innovation_spider(
            desirability=round(dvf_score.get("desirability_score") if isinstance(dvf_score.get("desirability_score"), (int, float)) else 50, 2),
            viability=round(dvf_score.get("viability_score") if isinstance(dvf_score.get("viability_score"), (int, float)) else 50, 2),
            feasibility=round(dvf_score.get("feasibility_score") if isinstance(dvf_score.get("feasibility_score"), (int, float)) else 50, 2),
            social_sentiment=round(reviews_social_summary.get("avg_social_sentiment") if isinstance(reviews_social_summary.get("avg_social_sentiment"), (int, float)) else 50, 2),
            reviews_rating=round(reviews_social_summary.get("avg_rating") if isinstance(reviews_social_summary.get("avg_rating"), (int, float)) else 3, 2),
        )

        score_summary(
            desirability=dvf_score.get("desirability_score") if isinstance(dvf_score.get("desirability_score"), (int, float)) else 50,
            viability=dvf_score.get("viability_score") if isinstance(dvf_score.get("viability_score"), (int, float)) else 50,
            feasibility=dvf_score.get("feasibility_score") if isinstance(dvf_score.get("feasibility_score"), (int, float)) else 50,
        )

#------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

col_info, col_dl_button = st.columns([7,1])
with col_info:
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
with col_dl_button:
    components.html("""
    <script>
    function printPage() {
        window.parent.print();
    }
    </script>

    <button onclick="printPage()" style="
        background: linear-gradient(135deg, #6b8fe0, #2b4fc9);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
    ">
    ⬇ Download Dashboard as PDF
    </button>
    """, height=70)


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
        filtered = run_filters(
            keywords,
            years_back=years_back,
            country=country
        )
        with st.spinner("Analyzing patents..."):
            patents_summary = summarize_patents(filtered["patents"])
        with st.spinner("Analyzing news..."):
            news_summary = summarize_news(filtered["news"])
        with st.spinner("Analyzing reviews and socials..."):
            reviews_social_summary = summarize_reviews_social(
            filtered["reviews"],
            filtered["social"]
            )
        with st.spinner("Analyzing catalogs..."):
            catalog_summary = summarize_catalog(filtered["catalog"])
        dvf_score = calculate_dvf_score(
            patents_summary=patents_summary,
            news_summary=news_summary,
            reviews_social_summary=reviews_social_summary,
            catalog_summary=catalog_summary,
        )

    st.session_state["filtered"] = filtered
    st.session_state["patents_summary"] = patents_summary
    st.session_state["news_summary"] = news_summary
    st.session_state["reviews_social_summary"] = reviews_social_summary
    st.session_state["catalog_summary"] = catalog_summary
    st.session_state["has_results"] = True
    st.session_state['dvf_score'] = dvf_score


if st.session_state.get("has_results"):
    patents_summary = st.session_state["patents_summary"]
    news_summary = st.session_state["news_summary"]
    reviews_social_summary = st.session_state["reviews_social_summary"]
    catalog_summary = st.session_state["catalog_summary"]
    dvf_score = st.session_state['dvf_score']

    st.markdown("<br>", unsafe_allow_html=True)
    render_patents_globe(st.session_state["filtered"]["patents"])

    st.markdown("<br>", unsafe_allow_html=True)