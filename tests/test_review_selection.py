"""Tests for top 3 critical review selection logic."""

import pandas as pd
import pytest

from src.review_selection import select_top_critical_reviews


def test_select_top_critical_reviews_ranking():
    # Construct DataFrame with varied ratings and review lengths
    df = pd.DataFrame({
        "Rating": [5, 1, 1, 2, 1, 1],
        "Title": ["Great", "Short complaint", "Longest complaint text", "Two stars", "Medium complaint text", "Very short"],
        "Review Text": [
            "This was wonderful and flattering in every way imaginable.",     # 5 stars - ignored
            "Terrible shirt.",                                                # 15 chars
            "This dress had terrible seams, awful fabric, completely itchy and see through.", # 78 chars
            "Did not fit well.",                                              # 2 stars - ignored
            "Awful material, buttons fell off immediately upon opening.",      # 58 chars
            "Bad.",                                                           # 4 chars
        ],
        "clean_text": ["clean1", "clean2", "clean3", "clean4", "clean5", "clean6"],
        "Clothing ID": [1, 2, 3, 4, 5, 6],
    })

    top_3 = select_top_critical_reviews(df, n=3)

    assert top_3.count == 3
    # All must have Rating == 1
    assert all(r.rating == 1 for r in top_3.reviews)

    # Must be ordered strictly descending by length: 78 -> 58 -> 15
    lengths = [r.review_length for r in top_3.reviews]
    assert lengths == [78, 58, 15]
    assert top_3.reviews[0].title == "Longest complaint text"
    assert top_3.reviews[1].title == "Medium complaint text"
    assert top_3.reviews[2].title == "Short complaint"


def test_select_top_critical_reviews_insufficient_records():
    df = pd.DataFrame({
        "Rating": [1, 2, 3],
        "Title": ["T1", "T2", "T3"],
        "Review Text": ["R1", "R2", "R3"],
        "clean_text": ["c1", "c2", "c3"],
    })
    # Only one 1-star review; requesting 3 should raise ValueError
    with pytest.raises(ValueError) as exc:
        select_top_critical_reviews(df, n=3)
    assert "fewer than the requested 3 reviews" in str(exc.value)
