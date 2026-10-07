"""Data cleaning and normalization module for customer reviews."""

import re
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd

from src.schemas import CleaningMetrics


def clean_review_text(text: Any) -> str:
    """Normalize raw review text for keyword and n-gram analysis.

    Preserves lowercase English letters and whitespace; strips out numbers,
    punctuation, and excess whitespace.
    """
    if pd.isna(text):
        return ""
    text_str = str(text).lower()
    text_str = re.sub(r"[^a-zA-Z\s]", " ", text_str)
    text_str = re.sub(r"\s+", " ", text_str)
    return text_str.strip()


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the customer reviews CSV dataset from disk."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path.resolve()}. "
            "Please ensure the CSV file is located in the data/ directory."
        )
    return pd.read_csv(csv_path)


def clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningMetrics]:
    """Execute standard deterministic data cleaning pipeline.

    Rules applied:
    1. Remove 'Unnamed: 0' index artifact if present.
    2. Convert 'Rating' to numeric, coercing invalid values to NaN.
    3. Treat whitespace-only review text as missing (NA).
    4. Drop rows with missing 'Review Text' or 'Rating'.
    5. Impute missing 'Title' with empty string.
    6. Validate that ratings strictly lie in the range 1-5.
    7. Remove exact duplicate rows.
    8. Generate a 'clean_text' column while strictly preserving original 'Review Text'.

    Returns:
        Tuple containing the cleaned DataFrame and CleaningMetrics.
    """
    rows_before = len(df_raw)

    # 1. Drop index column artifact
    df = df_raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()

    # 2. Convert Rating to numeric
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # 3. Treat whitespace-only Review Text as missing
    df["Review Text"] = df["Review Text"].replace(r"^\s*$", pd.NA, regex=True)

    # Track missing before drop
    missing_text_count = int(df["Review Text"].isna().sum())
    missing_rating_count = int(df["Rating"].isna().sum())

    # 4. Drop missing Review Text and Rating
    df = df.dropna(subset=["Review Text", "Rating"]).copy()

    # 5. Impute missing Title
    df["Title"] = df["Title"].fillna("")

    # 6. Validate rating boundaries 1-5
    df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)].copy()
    df["Rating"] = df["Rating"].astype(int)

    # 7. Deduplicate exact duplicate rows
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    # 8. Create clean_text column for NLP/keyword calculations
    df["clean_text"] = df["Review Text"].apply(clean_review_text)

    # Calculate review length for transparency
    df["review_length"] = df["Review Text"].str.len()

    rows_after = len(df)
    rows_removed = rows_before - rows_after

    metrics = CleaningMetrics(
        rows_before=rows_before,
        rows_after=rows_after,
        rows_removed=rows_removed,
        missing_review_text_dropped=missing_text_count,
        missing_ratings_dropped=missing_rating_count,
        duplicates_removed=int(duplicates_before),
    )

    return df, metrics
