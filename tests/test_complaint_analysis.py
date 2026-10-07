"""Tests for complaint keyword analysis functions."""

import pandas as pd
from src.complaint_analysis import get_top_keywords, get_top_bigrams, get_predefined_term_counts, get_keyword_insights


def make_critical_df():
    return pd.DataFrame({
        "reviewText": [
            "The fabric was terrible and see-through. Awful fit.",
            "Returned it immediately. Poor color faded fast.",
            "Way too small. Cheap material. Very itchy fabric.",
        ],
        "overall": [1, 2, 1],
        "clean_text": [
            "the fabric was terrible and see through awful fit",
            "returned it immediately poor color faded fast",
            "way too small cheap material very itchy fabric",
        ],
    })


def test_get_top_keywords_returns_list():
    df = make_critical_df()
    result = get_top_keywords(df, n=5)
    assert isinstance(result, list)
    assert len(result) <= 5


def test_get_top_keywords_has_term_count():
    df = make_critical_df()
    result = get_top_keywords(df)
    assert "term" in result[0]
    assert "count" in result[0]


def test_get_top_keywords_fabric_appears():
    """'fabric' appears twice so it should be in the top keywords."""
    df = make_critical_df()
    result = get_top_keywords(df, n=15)
    terms = [r["term"] for r in result]
    assert "fabric" in terms


def test_get_top_bigrams_returns_list():
    df = make_critical_df()
    result = get_top_bigrams(df, n=5)
    assert isinstance(result, list)


def test_get_top_bigrams_has_phrase_count():
    df = make_critical_df()
    result = get_top_bigrams(df)
    if result:
        assert "phrase" in result[0]
        assert "count" in result[0]


def test_get_predefined_term_counts_all_terms_present():
    from src.config import COMPLAINT_TERMS
    df = make_critical_df()
    result = get_predefined_term_counts(df)
    result_terms = [r["term"] for r in result]
    for term in COMPLAINT_TERMS:
        assert term in result_terms


def test_get_predefined_term_counts_fabric():
    df = make_critical_df()
    result = get_predefined_term_counts(df)
    fabric_entry = next((r for r in result if r["term"] == "fabric"), None)
    assert fabric_entry is not None
    assert fabric_entry["count"] == 2


def test_get_predefined_term_counts_see_through():
    df = make_critical_df()
    result = get_predefined_term_counts(df)
    see_through = next((r for r in result if r["term"] == "see-through"), None)
    assert see_through is not None
    assert see_through["count"] >= 1


def test_get_predefined_term_counts_sorted():
    """Results should be sorted descending by count."""
    df = make_critical_df()
    result = get_predefined_term_counts(df)
    counts = [r["count"] for r in result]
    assert counts == sorted(counts, reverse=True)


def test_get_keyword_insights_structure():
    df = make_critical_df()
    result = get_keyword_insights(df)
    assert "top_keywords" in result
    assert "predefined_terms" in result
    assert "top_bigrams" in result
