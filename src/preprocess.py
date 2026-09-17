import re

import numpy as np
import pandas as pd


def parse_amount(x):
    """Convert values like '42 Lac' / '1.40 Cr' to rupees."""
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    if "lac" in x:
        return float(x.replace("lac", "").strip()) * 100_000
    if "cr" in x:
        return float(x.replace("cr", "").strip()) * 10_000_000

    try:
        return float(x)
    except ValueError:
        return np.nan


def parse_area(x):
    """Convert area strings to square feet."""
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()
    match = re.search(r"([\d.]+)\s*([a-z]+)", x)
    if not match:
        try:
            return float(x)
        except ValueError:
            return np.nan

    value = float(match.group(1))
    unit = match.group(2)

    if unit in ["sqft", "sq.ft", "sq-ft"]:
        return value
    if unit == "sqm":
        return value * 10.7639
    if unit == "acre":
        return value * 43560
    if unit == "sqyrd":
        return value * 9
    if unit == "sqin":
        return value / 144
    return value


def parse_floor_current(x):
    """From '10 out of 11' → 10. 'Ground out of 7' → 0."""
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()
    if x.startswith("ground"):
        return 0

    match = re.search(r"^(\d+)", x)
    if match:
        return int(match.group(1))
    return np.nan


def parse_floor_total(x):
    """From '10 out of 11' → 11."""
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    match = re.search(r"out of\s*(\d+)", x)
    if match:
        return int(match.group(1))

    match = re.search(r"^(\d+)$", x)
    if match:
        return int(match.group(1))
    return np.nan


def load_and_clean_data(path: str, target: str = "amount(in rupees)") -> pd.DataFrame:
    """Load the raw CSV and apply the notebook cleaning steps."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    df[target] = df[target].str.lower().apply(parse_amount)

    q99 = df[target].quantile(0.99)
    df = df[df[target] <= q99].copy()

    df["bathroom"] = df["bathroom"].replace("> 10", 11)
    df["bathroom"] = pd.to_numeric(df["bathroom"], errors="coerce")

    df = df.drop(
        [
            "status",
            "index",
            "title",
            "price (in rupees)",
            "description",
            "overlooking",
            "society",
            "plot area",
            "dimensions",
            "ownership",
        ],
        axis=1,
    )

    df["balcony"] = df["balcony"].replace("> 10", 11)
    df["balcony"] = pd.to_numeric(df["balcony"], errors="coerce")

    df["car parking"] = df["car parking"].str.lower().str.strip()
    df["car covered"] = df["car parking"].str.contains("covered", na=False).astype(int)
    df["parking count"] = df["car parking"].str.count(",") + 1
    df["parking count"] = df["parking count"].fillna(0)

    furnishing_map = {
        "Unfurnished": 0,
        "Semi-Furnished": 1,
        "Furnished": 2,
    }
    df["furnishing"] = df["furnishing"].map(furnishing_map)

    df = df[df["transaction"].isin(["New Property", "Resale"])]
    df["transaction"] = np.where(df["transaction"] == "New Property", 1, 0)

    df["super_area_sqft"] = df["super area"].apply(parse_area)
    df["carpet_area_sqft"] = df["carpet area"].apply(parse_area)

    df = df[
        (df["carpet_area_sqft"].between(100, 20000)) | (df["carpet_area_sqft"].isna())
    ]
    df = df[
        (df["super_area_sqft"].between(100, 5000)) | (df["super_area_sqft"].isna())
    ]

    df = df.drop(
        ["super area", "carpet area"],
        axis=1,
    )
    df = df.drop("car parking", axis=1)

    df["facing"] = df["facing"].fillna("missing")
    df["facing"] = df["facing"].str.lower()

    df["floor_current"] = df["floor"].apply(parse_floor_current)
    df["floor_total"] = df["floor"].apply(parse_floor_total)

    mask = df["floor"].notna() & df["floor_total"].isna()
    df = df[~mask]
    df = df.drop("floor", axis=1)

    return df
