"""Tests for top 3 review selection logic."""

import pandas as pd
import pytest
from src.review_selection import select_top_3_reviews


def make_df():
    return pd.DataFrame({
        "reviewText": [
            "Short review.",
            "A much longer review with more specific complaints about the fabric and fit of this dress.",
            "Another medium length review about the poor quality of the item received.",
            "The longest review here — absolutely terrible experience, the dress was see-through, fabric was cheap, fit was off, and color looked nothing like the picture. Will never order again.",
            "Average review.",
        ],
        "overall": [1, 1, 1, 1, 3],
        "summary": ["Bad", "Detailed", "Medium", "Worst ever", "Okay"],
        "asin": ["A1", "A2", "A3", "A4", "A5"],
        "helpful": [0, 5, 2, 10, 1],
        "clean_text": [
            "short review",
            "much longer review specific complaints fabric fit dress",
            "medium length review poor quality item received",
            "longest review absolutely terrible experience dress see through fabric cheap fit off color nothing picture",
            "average review",
        ],
    })


def test_select_top_3_returns_3():
    df = make_df()
    result = select_top_3_reviews(df)
    assert len(result) == 3


def test_select_top_3_all_lowest_rating():
    df = make_df()
    result = select_top_3_reviews(df)
    min_rating = df["overall"].min()
    for review in result:
        assert review["overall"] == min_rating


def test_select_top_3_sorted_by_length():
    """The longest review should be first."""
    df = make_df()
    result = select_top_3_reviews(df)
    lengths = [r["review_length"] for r in result]
    assert lengths == sorted(lengths, reverse=True)


def test_select_top_3_has_required_keys():
    df = make_df()
    result = select_top_3_reviews(df)
    required_keys = {"index", "asin", "overall", "summary", "reviewText", "helpful", "review_length", "detected_complaints"}
    for review in result:
        assert required_keys.issubset(review.keys())


def test_select_top_3_detected_complaints():
    """The longest review should detect some complaint keywords."""
    df = make_df()
    result = select_top_3_reviews(df)
    # The longest review has 'fabric', 'fit', 'see-through', 'color'
    assert len(result[0]["detected_complaints"]) > 0


def test_select_top_3_raises_if_not_enough():
    df = pd.DataFrame({
        "reviewText": ["Short", "Also short"],
        "overall": [1, 1],
        "summary": ["A", "B"],
        "asin": ["X", "Y"],
        "helpful": [0, 0],
        "clean_text": ["short", "also short"],
    })
    with pytest.raises(ValueError):
        select_top_3_reviews(df)
