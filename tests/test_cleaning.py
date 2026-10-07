"""Tests for data cleaning functions."""

import pandas as pd
import pytest
from src.cleaning import clean_text, normalize_columns, validate_columns, clean_dataset
from src.config import REQUIRED_COLUMNS


def make_sample_df():
    """Create a small sample DataFrame with the required 5 columns."""
    return pd.DataFrame({
        "reviewText": [
            "The fabric is amazing and fits perfectly!",
            "Very poor quality, it was see-through and too small.",
            "   ",  # whitespace-only, should be dropped
            "The color faded after one wash.",
        ],
        "overall": [5, 1, 3, 2],
        "summary": ["Great dress", "Terrible quality", "Okay", "Disappointing"],
        "asin": ["A1", "A2", "A3", "A1"],
        "helpful": [10, 5, 0, 3],
    })


def test_clean_text_basic():
    assert clean_text("Hello, World! 123") == "hello world"


def test_clean_text_none():
    assert clean_text(None) == ""


def test_clean_text_extra_spaces():
    result = clean_text("  too   many   spaces  ")
    assert "  " not in result


def test_normalize_columns_kaggle_format():
    """Test that Kaggle-format column names are renamed correctly."""
    kaggle_df = pd.DataFrame({
        "Review Text": ["Nice dress"],
        "Rating": [5],
        "Title": ["Love it"],
        "Clothing ID": ["C1"],
        "Positive Feedback Count": [2],
    })
    result = normalize_columns(kaggle_df)
    assert "reviewText" in result.columns
    assert "overall" in result.columns
    assert "summary" in result.columns
    assert "asin" in result.columns
    assert "helpful" in result.columns


def test_normalize_columns_already_correct():
    """Test that already-correct columns are untouched."""
    df = make_sample_df()
    result = normalize_columns(df)
    assert list(df.columns) == list(result.columns)


def test_validate_columns_pass():
    df = make_sample_df()
    ok, missing = validate_columns(df)
    assert ok is True
    assert missing == []


def test_validate_columns_fail():
    df = pd.DataFrame({"reviewText": ["text"], "overall": [5]})
    ok, missing = validate_columns(df)
    assert ok is False
    assert "summary" in missing
    assert "asin" in missing
    assert "helpful" in missing


def test_clean_dataset_removes_missing_text():
    df = make_sample_df()
    cleaned, summary = clean_dataset(df)
    # The whitespace-only row should be dropped
    assert len(cleaned) < len(df)
    assert summary["rows_removed"] > 0


def test_clean_dataset_overall_is_int():
    df = make_sample_df()
    cleaned, _ = clean_dataset(df)
    assert cleaned["overall"].dtype in [int, "int64", "int32"]


def test_clean_dataset_ratings_valid_range():
    df = make_sample_df()
    cleaned, _ = clean_dataset(df)
    assert cleaned["overall"].between(1, 5).all()


def test_clean_dataset_has_clean_text():
    df = make_sample_df()
    cleaned, _ = clean_dataset(df)
    assert "clean_text" in cleaned.columns


def test_clean_dataset_invalid_schema():
    bad_df = pd.DataFrame({"text": ["hi"], "score": [5]})
    with pytest.raises(ValueError, match="Dataset is missing required columns"):
        clean_dataset(bad_df)
