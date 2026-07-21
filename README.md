# SDA-Innovation-Studio
Small Domestic Appliances dashboard for managerial advices based on multi-source dataset

A Python-based web app that combines patents, news, user reviews, and product catalog data to help an innovation team discover and compare future product opportunities in Small Domestic Appliances (SDA) — for example, air purifiers, air fryers, coffee machines, and robot vacuums.

## Overview

This tool lets a non-technical innovation or product manager select an opportunity theme (e.g. *"Smart, timer-equipped espresso machine with minimal maintenance procedures and low ongoing costs"*), and view an evidence-backed summary of demand and supply signals, along with a simple Desirability–Viability–Feasibility (DVF) assessment.

## Features

- **Theme-based exploration**: Choose from pre-defined SDA opportunity themes.
- **Multi-source evidence view**: Patents, news, reviews, and product catalog signals shown side by side.
- **Opportunity cards**: Auto-generated demand/supply bullet summaries and DVF scores per theme.
- **Update & history tracking**: Refresh dynamic data (patents, news) on demand; every update is saved as a timestamped snapshot so changes can be tracked over time.

## Project Structure

SDA-INNOVATION-STUDIO/
├── data/
│   ├── clean/
│   ├── raw/
│   └── static/
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── score_summary.html
│   │   ├── score_summary.py
│   │   ├── spider_chart.html
│   │   └── spider_chart.py
│   ├── data_builder/
│   │   ├── catalog_clean_fe.py
│   │   ├── news_clean_fe.py
│   │   ├── news_ingest.py
│   │   ├── patents_clean_fe.py
│   │   ├── patents_ingest.py
│   │   ├── reviews_clean_fe.py
│   │   ├── reviews_ingest.py
│   │   ├── social_clean_fe.py
│   │   └── social_ingest.py
│   ├── pipeline/
│   │   ├── analyzer.py
│   │   ├── cleaner.py
│   │   └── nlp_analyzer.py
│   └── prompt_reader.py
├── .env
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── ui.py

## Data Sources

| Source | Type | Fields |
|---|---|---|
| EPO Open Patent Services (OPS) | Patents | publication number, title, abstract, applicant, publication date, CPC classes |
| News API / dataset | News | title, description, publication date, source, URL |
| Kaggle review dataset(s) | Reviews | review text, rating, timestamp, product metadata |
| Synthetic appliance catalog | Product catalog | Company, Brand, Appliance, Energy Consumption, SKU |
| Reddit (optional) | Social | subreddit, score, date, title, text |

## Data Pipeline

1. **Ingestion**: Pull raw data from APIs/datasets and store as CSV in `data/raw/<timestamp>/`.
2. **Cleaning**: Fix missing/malformed rows, normalize dates and text, remove duplicates.
3. **Feature Engineering**: Derive columns (publication year, sentiment, keyword flags, SDA category, energy bands, etc.) and save to `data/clean/<timestamp>/`.
4. **Theme Aggregation**: Map cleaned records to opportunity themes using keyword rules; compute theme-level metrics (patent volume trends, sentiment distributions, catalog coverage, DVF scores).
5. **Web App**: Streamlit reads the latest (or a selected historical) snapshot and renders the evidence view and opportunity cards.

## Opportunity Themes (Examples)

- Smart, timer-equipped espresso machine with minimal maintenance procedures and low ongoing costs.
- Floor-adaptable robot vacuum cleaner with liquid safety.
- Energy-efficient air fryer that cooks quickly while minimizing heat buildup in the room.

## Setup & Installation

Create a `.env` file with your API credentials:
PATENT_CUSTOMER_KEY=your_epo_ops_key
PATENT_CUSTOMER_SECRET_KEY=your_epo_ops_secret
NEWS_API_KEY=your_news_api_key

## Running the App

```bash
streamlit run ui.py
```

## Opportunity Card & DVF Scoring

## Limitations

- Free-tier API limits may restrict data volume and features like time-window filtering.
- Hardware constraints may prevent using a pre-trained model for smarter results.

## Future Work

- Add an "Update" button to refresh news and patent data on demand. *(implemented / in progress)*
- Log user activity to power trending signals over time.
- Cluster users by preferences to send targeted category updates.

## Tech Stack

- **Web app**: Streamlit
- **Data processing**: pandas, numpy, requests
- **NLP**: scikit-learn / spaCy / NLTK / TextBlob
- **Visualization**: Plotly / Altair

## License

Internal research/demo tool — not intended for production use.

## Link to the deployed dashboard:
https://sda-innovation-studio.streamlit.app/

## Link to the presentation:
https://docs.google.com/presentation/d/16ySUaVORkimo2DZzFa-I29qlflRfRXHj4Y2cnuHGfZM/edit?usp=sharing
