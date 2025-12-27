import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = "data/housing.csv"
REPORTS_DIR = "outputs/reports"
PLOTS_DIR = "outputs/plots"
CLEAN_PATH = "outputs/cleaned_housing.csv"

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_dates(df):
    # Try to parse Date if exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # Extract year, month if helpful
        df["SaleYear"] = df["Date"].dt.year
        df["SaleMonth"] = df["Date"].dt.month
    return df

def clean_data(df):
    # Remove duplicates by id if present
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])

    # Strip and standardize column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

    # Expected columns mapping after standardization
    expected = [
        "id","date","number_of_bedrooms","number_of_bathrooms","living_area","lot_area",
        "number_of_floors","waterfront_present","number_of_views","condition_of_the_house",
        "grade_of_the_house","area_of_the_house_excluding_basement","area_of_the_basement",
        "built_year","renovation_year","postal_code","lattitude","longitude","living_area_renov",
        "lot_area_renov","number_of_schools_nearby","distance_from_the_airport","price"
    ]

    # Ensure types where sensible
    num_cols = [c for c in df.columns if c not in ["date","postal_code"]]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean postal_code as categorical/string if exists
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype(str).str.strip()

    # Handle missing values: simple strategy
    # - Numeric: median
    # - Categorical: mode
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "unknown")

    # Make waterfront_present binary if not already (assume 1/0 or yes/no)
    if "waterfront_present" in df.columns and df["waterfront_present"].dtype == object:
        df["waterfront_present"] = df["waterfront_present"].str.lower().map({"yes":1,"no":0}).fillna(0)

    # Derived features (simple examples)
    if {"built_year","renovation_year"}.issubset(df.columns):
        df["house_age"] = (df["sale_year"] if "sale_year" in df.columns else pd.Timestamp.now().year) - df["built_year"]
        df["years_since_renov"] = np.where(df["renovation_year"]>0,
                                           ((df["sale_year"] if "sale_year" in df.columns else pd.Timestamp.now().year) - df["renovation_year"]),
                                           df["house_age"])

    return df

def eda(df):
    # Save basic description
    desc = df.describe(include="all")
    desc.to_csv(os.path.join(REPORTS_DIR, "describe.csv"))

    # Correlation heatmap (numeric)
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12,8))
    sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "corr_heatmap.png"))
    plt.close()

    # Price distributions
    if "price" in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df["price"], kde=True)
        plt.title("Price distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "price_distribution.png"))
        plt.close()

    # Scatter examples: living_area vs price
    if {"living_area","price"}.issubset(df.columns):
        plt.figure(figsize=(8,5))
        sns.scatterplot(x="living_area", y="price", data=df, alpha=0.5)
        plt.title("Living area vs Price")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "living_area_vs_price.png"))
        plt.close()

def main():
    ensure_dirs()
    df = pd.read_csv(RAW_PATH)
    df = parse_dates(df)
    df = clean_data(df)
    eda(df)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_PATH}")

if __name__ == "__main__":
    main()
