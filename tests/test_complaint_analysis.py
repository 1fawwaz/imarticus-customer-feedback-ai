"""Tests for complaint keywords, bigrams, and predefined term counts."""

import pandas as pd
import pytest

from src.complaint_analysis import (
    get_top_complaint_keywords,
    get_top_bigrams,
    get_predefined_term_counts,
)


def test_complaint_keywords_and_bigrams():
    sample_df = pd.DataFrame({
        "Review Text": [
            "Poor quality fabric. Very cheap material.",
            "It was too small and see through.",
            "Completely see through dress, poor quality.",
        ],
        "clean_text": [
            "poor quality fabric very cheap material",
            "it was too small and see through",
            "completely see through dress poor quality",
        ]
    })

    top_kw = get_top_complaint_keywords(sample_df, n=10)
    kw_terms = [item.term for item in top_kw]
    assert "quality" in kw_terms or "poor" in kw_terms or "fabric" in kw_terms

    top_bg = get_top_bigrams(sample_df, n=10)
    phrases = [item.phrase for item in top_bg]
    assert "poor quality" in phrases
    assert "see through" in phrases


def test_predefined_terms_handles_see_through():
    df = pd.DataFrame({
        "Review Text": [
            "Fabric is see-through in the sun.",
            "The top is totally see through and cheap.",
            "Returned because of poor color and fit.",
        ],
        "clean_text": [
            "fabric is see through in the sun",
            "the top is totally see through and cheap",
            "returned because of poor color and fit",
        ]
    })

    term_counts = {item.term: item.count for item in get_predefined_term_counts(df)}
    assert term_counts["see-through"] == 2
    assert term_counts["cheap"] == 1
    assert term_counts["returned"] == 1
    assert term_counts["fit"] == 1
    assert term_counts["color"] == 1
