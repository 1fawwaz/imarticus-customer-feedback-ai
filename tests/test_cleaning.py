"""Tests for data cleaning logic."""

import pandas as pd
import numpy as np
import pytest

from src.cleaning import clean_review_text, clean_dataset


def test_clean_review_text():
    raw_sample = "This dress was TOO SMALL!! And poor quality... 123 stars!!"
    cleaned = clean_review_text(raw_sample)
    assert "too small" in cleaned
    assert "poor quality" in cleaned
    assert "123" not in cleaned
    assert "!" not in cleaned
    assert cleaned == "this dress was too small and poor quality stars"


def test_clean_dataset_handles_missing_and_duplicates():
    raw_data = {
        "Unnamed: 0": [0, 1, 2, 3, 4, 5],
        "Clothing ID": [101, 102, 103, 104, 105, 105],
        "Age": [30, 40, 25, 50, 35, 35],
        "Title": [None, "Great fit", "Bad fabric", "", "Duplicate", "Duplicate"],
        "Review Text": [
            "Nice color but too tight.",
            np.nan,                      # Missing review text
            "   ",                       # Whitespace review text
            "Terrible stitching.",
            "Loved it completely.",
            "Loved it completely."       # Exact duplicate
        ],
        "Rating": [2, 5, 1, "invalid", 4, 4],  # Invalid rating included
        "Recommended IND": [0, 1, 0, 0, 1, 1],
        "Positive Feedback Count": [0, 2, 1, 0, 3, 3],
        "Division Name": ["General", "General", "General", "General", "General", "General"],
        "Department Name": ["Dresses", "Dresses", "Tops", "Tops", "Dresses", "Dresses"],
        "Class Name": ["Dresses", "Dresses", "Blouses", "Blouses", "Dresses", "Dresses"],
    }
    df_raw = pd.DataFrame(raw_data)
    df_clean, metrics = clean_dataset(df_raw)

    # Verifications
    assert "Unnamed: 0" not in df_clean.columns
    assert "clean_text" in df_clean.columns
    assert "review_length" in df_clean.columns

    # Rows with missing text, whitespace text, or invalid rating must be dropped
    assert len(df_clean) == 2  # Only row 0 (rating 2) and row 4 (rating 4) remain after deduplication
    assert metrics.rows_before == 6
    assert metrics.rows_after == 2
    assert metrics.rows_removed == 4
    assert metrics.duplicates_removed == 1

    # Missing Title should be imputed with empty string
    assert df_clean.iloc[0]["Title"] == ""

    # Ratings must be integers between 1 and 5
    assert all(df_clean["Rating"].isin([1, 2, 3, 4, 5]))
