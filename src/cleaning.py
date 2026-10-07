"""Data cleaning functions for the customer review dataset."""

import re
from pathlib import Path
from typing import Tuple, List
import pandas as pd

from src.config import REQUIRED_COLUMNS


def clean_text(text) -> str:
    """Convert raw review text to lowercase letters only (for keyword analysis)."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename official Kaggle dataset columns to the project's standard column names.
    
    The Kaggle CSV uses different column names (e.g., 'Review Text' instead of
    'reviewText'). This maps them to the 5 standard columns used throughout the project.
    """
    if "Review Text" in df.columns and "reviewText" not in df.columns:
        rename_map = {
            "Review Text": "reviewText",
            "Rating": "overall",
            "Title": "summary",
            "Clothing ID": "asin",
            "Positive Feedback Count": "helpful",
        }
        cols_present = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df[list(cols_present.keys())].rename(columns=cols_present)
        # Add missing columns with defaults
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col == "summary" else 0
    return df


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the reviews CSV from disk and normalize column names."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Make sure 'Womens Clothing E-Commerce Reviews.csv' is in the data/ folder."
        )
    df = pd.read_csv(csv_path)
    return normalize_columns(df)


def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check that all 5 required columns are present in the DataFrame."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Clean the raw reviews DataFrame.

    Steps:
    1. Normalize column names (handle Kaggle format).
    2. Validate required columns are present.
    3. Convert 'overall' (rating) to numeric.
    4. Drop rows with missing review text or rating.
    5. Fill missing 'summary' with empty string.
    6. Keep only valid ratings (1 to 5).
    7. Remove exact duplicate rows.
    8. Add 'clean_text' column for keyword analysis.
    
    Returns the cleaned DataFrame and a summary dict of what was removed.
    """
    df_raw = normalize_columns(df_raw)

    is_valid, missing = validate_columns(df_raw)
    if not is_valid:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing)}\n"
            f"Required: {REQUIRED_COLUMNS}"
        )

    rows_before = len(df_raw)
    df = df_raw.copy()

    # Convert rating to numeric
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

    # Treat whitespace-only review text as missing
    df["reviewText"] = df["reviewText"].replace(r"^\s*$", pd.NA, regex=True)

    missing_text = int(df["reviewText"].isna().sum())
    missing_rating = int(df["overall"].isna().sum())

    # Drop rows missing review text or rating
    df = df.dropna(subset=["reviewText", "overall"]).copy()

    # Fill missing summary
    df["summary"] = df["summary"].fillna("")

    # Keep only ratings 1–5
    df = df[(df["overall"] >= 1) & (df["overall"] <= 5)].copy()
    df["overall"] = df["overall"].astype(int)

    # Remove duplicates
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Add clean text for keyword analysis
    df["clean_text"] = df["reviewText"].apply(clean_text)

    rows_after = len(df)

    cleaning_summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "missing_review_text": missing_text,
        "missing_rating": missing_rating,
        "duplicates_removed": duplicates,
    }

    return df, cleaning_summary
