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
