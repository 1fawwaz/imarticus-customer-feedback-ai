"""Tests for critical review filtering and dataset analysis."""

import pandas as pd
import pytest

from src.analysis import filter_critical_reviews, get_dataset_overview, detect_complaint_indicators


def test_filter_critical_reviews_rule_based():
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5, 2, 1],
        "Review Text": ["rev1", "rev2", "rev3", "rev4", "rev5", "rev6", "rev7"],
        "clean_text": ["rev1", "rev2", "rev3", "rev4", "rev5", "rev6", "rev7"],
    })
    critical_df = filter_critical_reviews(df)

    # Strictly ratings 1 and 2
    assert len(critical_df) == 4
    assert set(critical_df["Rating"].unique()) == {1, 2}
    assert 3 not in critical_df["Rating"].values
    assert 4 not in critical_df["Rating"].values
    assert 5 not in critical_df["Rating"].values


def test_detect_complaint_indicators():
    text_1 = "The dress was way too small and the fabric felt cheap and itchy."
    indicators_1 = detect_complaint_indicators(text_1)
    assert "small" in indicators_1
    assert "fabric" in indicators_1
    assert "cheap" in indicators_1
    assert "itchy" in indicators_1

    text_2 = "Fabric was see through and poor quality."
    indicators_2 = detect_complaint_indicators(text_2)
    assert "see-through" in indicators_2
    assert "poor" in indicators_2


def test_filter_critical_reviews_default():
    """filter_critical_reviews(df) without threshold must default to 2."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Review Text": ["terrible", "bad", "okay", "good", "great"]
    })
    res = filter_critical_reviews(df)
    assert len(res) == 2
    assert set(res["Rating"].tolist()) == {1, 2}


def test_filter_critical_reviews_threshold_2():
    """filter_critical_reviews(df, threshold=2) filters Rating <= 2."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Review Text": ["terrible", "bad", "okay", "good", "great"]
    })
    res = filter_critical_reviews(df, threshold=2)
    assert len(res) == 2
    assert set(res["Rating"].tolist()) == {1, 2}


def test_filter_critical_reviews_threshold_3():
    """filter_critical_reviews(df, threshold=3) filters Rating <= 3."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Review Text": ["terrible", "bad", "okay", "good", "great"]
    })
    res = filter_critical_reviews(df, threshold=3)
    assert len(res) == 3
    assert set(res["Rating"].tolist()) == {1, 2, 3}


def test_filter_critical_reviews_custom_2_to_5_dataset():
    """Custom dataset with ratings [2, 3, 4, 5]:
    threshold=2 -> 1 critical review (Rating 2).
    threshold=3 -> 2 critical reviews (Rating 2, 3).
    """
    df = pd.DataFrame({
        "Rating": [2, 3, 4, 5],
        "Review Text": ["bad", "okay", "good", "great"]
    })
    crit_2 = filter_critical_reviews(df, threshold=2)
    assert len(crit_2) == 1
    assert crit_2.iloc[0]["Rating"] == 2
    assert crit_2.iloc[0]["Review Text"] == "bad"

    crit_3 = filter_critical_reviews(df, threshold=3)
    assert len(crit_3) == 2
    assert set(crit_3["Rating"].tolist()) == {2, 3}


def test_filter_critical_reviews_standard_1_to_5_dataset():
    """Standard 1-5 scale: threshold=2 yields 1 and 2, excludes 3, 4, 5."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Review Text": ["terrible", "bad", "average", "good", "great"]
    })
    crit = filter_critical_reviews(df, threshold=2)
    assert len(crit) == 2
    assert set(crit["Rating"].tolist()) == {1, 2}
    assert 3 not in crit["Rating"].values
    assert 4 not in crit["Rating"].values
    assert 5 not in crit["Rating"].values


def test_filter_critical_reviews_missing_rating_column():
    """Missing Rating column raises ValueError with descriptive message."""
    df = pd.DataFrame({
        "Review Text": ["bad", "good"]
    })
    with pytest.raises(ValueError, match="Rating"):
        filter_critical_reviews(df)


def test_filter_critical_reviews_non_numeric_rating():
    """Non-numeric Rating values raise ValueError."""
    df = pd.DataFrame({
        "Rating": ["bad", "terrible", "poor"],
        "Review Text": ["a", "b", "c"]
    })
    with pytest.raises(ValueError, match="non-numeric"):
        filter_critical_reviews(df)


def test_filter_critical_reviews_preserves_original_df():
    """Filtering must not mutate original DataFrame in place."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Review Text": ["r1", "r2", "r3", "r4", "r5"]
    })
    df_copy = df.copy(deep=True)
    res = filter_critical_reviews(df, threshold=2)
    # Check original is untouched
    pd.testing.assert_frame_equal(df, df_copy)
    assert len(res) == 2

