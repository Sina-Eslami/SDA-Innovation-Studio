import pandas as pd
import numpy as np

from pathlib import Path

import re

BASE_DIR = Path(__file__).resolve().parent
file_dir = BASE_DIR / ".." / ".." / "data" / "static"

catalog_df = pd.read_csv(file_dir / 'raw - catalogs.csv')


def clean_catalog_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in ["Company", "Brand", "Appliance", "SKU"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
            df[col] = df[col].replace({"": np.nan, "None": np.nan, "nan": np.nan})

    df["SKU"] = df["SKU"].str.upper()
    df["Appliance"] = df["Appliance"].str.title()

    def parse_energy(val):
        if pd.isna(val):
            return pd.Series([np.nan, np.nan])
        s = str(val).strip().lower().replace(" ", "")
        m = re.match(r"([\d.]+)(kwh/year|kwh|kw|w)", s)
        if not m:
            return pd.Series([np.nan, np.nan])
        num, unit = float(m.group(1)), m.group(2)
        if unit == "kwh/year":
            return pd.Series([num, "kWh/year"])
        if unit == "kw":
            return pd.Series([num * 1000, "W"])
        if unit == "w":
            return pd.Series([num, "W"])
        return pd.Series([np.nan, np.nan])

    df[["ENERGY_VALUE", "ENERGY_UNIT"]] = df["Energy Consumption"].apply(parse_energy)
    df = df.drop(columns=["Energy Consumption"])

    df = df.dropna(subset=["SKU", "Appliance"])

    # 4. Deduplicate on SKU (unique product identifier)
    df = df.drop_duplicates(subset=["SKU"], keep="first")

    df["Company"] = df["Company"].fillna(df["Brand"])
    df["Company"] = df["Company"].fillna("Unknown")
    df["Brand"] = df["Brand"].fillna(df["Company"])

    return df.reset_index(drop=True)


def engineer_catalog_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    watt_mask = df["ENERGY_UNIT"] == "W"
    df["ENERGY_TIER"] = pd.cut(
        df.loc[watt_mask, "ENERGY_VALUE"],
        bins=[0, 500, 1000, 1500, np.inf],
        labels=["low", "medium", "high", "very_high"]
    )

    df["BRAND_PRODUCT_COUNT"] = df.groupby("Brand")["SKU"].transform("count")
    df["APPLIANCE_CATEGORY_COUNT"] = df.groupby("Appliance")["SKU"].transform("count")

    avg_energy_by_appliance = df[watt_mask].groupby("Appliance")["ENERGY_VALUE"].mean()
    df["AVG_ENERGY_FOR_APPLIANCE"] = df["Appliance"].map(avg_energy_by_appliance)
    df["ENERGY_VS_APPLIANCE_AVG"] = df["ENERGY_VALUE"] - df["AVG_ENERGY_FOR_APPLIANCE"]

    df["HAS_ENERGY_DATA"] = df["ENERGY_VALUE"].notna()
    df["IS_MULTI_SKU_APPLIANCE_LINE"] = df["APPLIANCE_CATEGORY_COUNT"] > 1

    return df