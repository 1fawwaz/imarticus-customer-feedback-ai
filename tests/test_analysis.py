"""Tests for review analysis functions."""

import pandas as pd
import pytest
from src.analysis import filter_critical_reviews, get_dataset_overview, detect_complaints, query_reviews


def make_df():
    """Sample cleaned DataFrame for testing."""
    return pd.DataFrame({
        "reviewText": [
            "Great dress, perfect fit!",
            "Terrible quality, see-through fabric.",
            "Okay, nothing special.",
            "Awful, returned immediately. Poor color.",
        ],
        "overall": [5, 1, 3, 2],
        "summary": ["Love it", "Terrible", "Fine", "Disappointed"],
        "asin": ["A1", "A2", "A3", "A1"],
        "helpful": [10, 5, 0, 3],
        "clean_text": [
            "great dress perfect fit",
            "terrible quality see through fabric",
            "okay nothing special",
            "awful returned immediately poor color",
        ],
    })


def test_filter_critical_reviews_count():
    df = make_df()
    result = filter_critical_reviews(df)
    assert len(result) == 2  # ratings 1 and 2


def test_filter_critical_reviews_ratings():
    df = make_df()
    result = filter_critical_reviews(df)
    assert result["overall"].max() <= 2


def test_filter_critical_reviews_empty():
    df = pd.DataFrame({
        "reviewText": ["Great!"],
        "overall": [5],
        "summary": ["Good"],
        "asin": ["A1"],
        "helpful": [0],
        "clean_text": ["great"],
    })
    result = filter_critical_reviews(df)
    assert len(result) == 0


def test_get_dataset_overview_structure():
    df = make_df()
    overview = get_dataset_overview(df, df)
    assert "total_raw_rows" in overview
    assert "cleaned_rows" in overview
    assert "critical_count" in overview
    assert "critical_pct" in overview
    assert "rating_distribution" in overview


def test_get_dataset_overview_critical_count():
    df = make_df()
    overview = get_dataset_overview(df, df)
    assert overview["critical_count"] == 2


def test_get_dataset_overview_critical_pct():
    df = make_df()
    overview = get_dataset_overview(df, df)
    assert overview["critical_pct"] == 50.0


def test_detect_complaints_basic():
    result = detect_complaints("The fabric was see-through and did not fit.")
    assert "fabric" in result
    assert "see-through" in result
    assert "fit" in result


def test_detect_complaints_no_match():
    result = detect_complaints("This was a great purchase, very happy!")
    assert result == []


def test_detect_complaints_see_through_space():
    """'see through' (without hyphen) should also be detected."""
    result = detect_complaints("The dress was see through when worn outside.")
    assert "see-through" in result


def test_query_reviews_critical_only():
    df = make_df()
    result = query_reviews(df, critical_only=True)
    assert len(result) == 2
    assert result["overall"].max() <= 2


def test_query_reviews_by_rating():
    df = make_df()
    result = query_reviews(df, rating=5)
    assert len(result) == 1
    assert result.iloc[0]["overall"] == 5


def test_query_reviews_keyword():
    df = make_df()
    result = query_reviews(df, critical_only=True, keyword="fabric")
    assert len(result) >= 1


def test_query_reviews_limit():
    df = make_df()
    result = query_reviews(df, limit=2)
    assert len(result) <= 2
