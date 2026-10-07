"""End-to-end pipeline test using a small sample dataset."""

import pandas as pd
from src.cleaning import clean_dataset, normalize_columns
from src.analysis import filter_critical_reviews, get_dataset_overview
from src.complaint_analysis import get_keyword_insights
from src.review_selection import select_top_3_reviews


def make_sample_df():
    """Create a minimal test dataset with the 5 required columns."""
    return pd.DataFrame({
        "reviewText": [
            "The fabric was see-through and the fit was terrible. Very disappointed.",
            "Great dress, loved the color and the fit was perfect for my size.",
            "Returned it. Poor quality fabric, itchy material, cheap looking color.",
            "Beautiful item, highly recommend to everyone!",
            "Way too small and the color was completely different from the picture.",
            "Awful experience, fabric was rough, fit was wrong, color was off. Never again.",
        ],
        "overall": [1, 5, 2, 5, 1, 1],
        "summary": ["Awful", "Great", "Returned", "Loved it", "Wrong size", "Terrible"],
        "asin": ["A1", "B2", "A1", "C3", "D4", "E5"],
        "helpful": [3, 8, 2, 5, 1, 4],
    })


def test_full_pipeline_runs():
    df_raw = make_sample_df()

    # Step 1: Clean
    df_clean, summary = clean_dataset(df_raw)
    assert len(df_clean) > 0
    assert "clean_text" in df_clean.columns
    assert summary["rows_removed"] >= 0

    # Step 2: Filter critical
    critical = filter_critical_reviews(df_clean)
    assert len(critical) == 4  # ratings 1, 2, 1, 1
    assert critical["overall"].max() <= 2

    # Step 3: Overview
    overview = get_dataset_overview(df_raw, df_clean)
    assert overview["total_raw_rows"] == 6
    assert overview["cleaned_rows"] == 6
    assert overview["critical_count"] == 4
    assert round(overview["critical_pct"], 2) == round(4 / 6 * 100, 2)

    # Step 4: Complaint analysis
    insights = get_keyword_insights(critical)
    assert "top_keywords" in insights
    assert "predefined_terms" in insights
    assert "top_bigrams" in insights

    # Check 'fabric' is detected
    term_dict = {r["term"]: r["count"] for r in insights["predefined_terms"]}
    assert term_dict["fabric"] >= 2

    # Step 5: Top 3 reviews
    top3 = select_top_3_reviews(df_clean)
    assert len(top3) == 3
    min_rating = df_clean["overall"].min()
    for r in top3:
        assert r["overall"] == min_rating


def test_rating_distribution():
    df_raw = make_sample_df()
    df_clean, _ = clean_dataset(df_raw)
    overview = get_dataset_overview(df_raw, df_clean)
    dist = overview["rating_distribution"]
    assert dist.get(1, 0) == 3   # three 1-star reviews
    assert dist.get(5, 0) == 2   # two 5-star reviews
    assert dist.get(2, 0) == 1   # one 2-star review
