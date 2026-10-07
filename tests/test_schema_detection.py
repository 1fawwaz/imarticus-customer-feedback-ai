"""Tests for smart CSV schema detection and column mapping.

Covers:
- Official Imarticus schema detection
- Lowercase column names
- Whitespace / underscore column names
- Review Text alias variations
- Rating alias variations
- Optional column absence (graceful)
- Unsupported dataset rejection
- Numeric rating validation
- Critical review filtering with threshold
- Top-3 selection with dynamic minimum rating
- Column mapping application
"""

import pandas as pd
import pytest

from src.schema_detection import detect_schema, apply_mapping, get_critical_threshold
from src.analysis import filter_critical_reviews
from src.review_selection import select_top_critical_reviews


# ---------------------------------------------------------------------------
# Helpers to build minimal test DataFrames
# ---------------------------------------------------------------------------

def _df(cols_and_values: dict) -> pd.DataFrame:
    n = max(len(v) for v in cols_and_values.values())
    data = {k: (v if len(v) == n else v * (n // len(v) + 1))[:n]
            for k, v in cols_and_values.items()}
    return pd.DataFrame(data)


def _review_df(**kwargs) -> pd.DataFrame:
    """Build a minimal review DataFrame with 5 rows."""
    review_col = kwargs.get("review_col", "Review Text")
    rating_col = kwargs.get("rating_col", "Rating")
    extra = kwargs.get("extra", {})
    data = {
        review_col: [
            "Fabric was very poor quality.",
            "Too small and itchy material.",
            "Great product, loved it!",
            "Returned immediately.",
            "Color faded after one wash.",
        ],
        rating_col: [1, 1, 5, 2, 2],
    }
    data.update(extra)
    return pd.DataFrame(data)


# ===========================================================================
# Schema A — Official / canonical column names
# ===========================================================================

def test_official_schema_detected():
    """Test A: exact official column names → compatible + is_official."""
    df = _review_df(extra={"Title": ["T1", "T2", "T3", "T4", "T5"],
                            "Clothing ID": [1, 2, 3, 4, 5],
                            "Age": [30, 25, 40, 35, 50],
                            "Recommended IND": [0, 0, 1, 0, 0],
                            "Positive Feedback Count": [0, 1, 5, 0, 2],
                            "Division Name": ["G", "G", "G", "G", "G"],
                            "Department Name": ["Tops", "Tops", "Dresses", "Bottoms", "Tops"],
                            "Class Name": ["Blouses", "Blouses", "Dresses", "Jeans", "Blouses"]})
    result = detect_schema(df)
    assert result.compatible is True
    assert result.is_official_dataset is True
    assert result.mapping is not None
    assert result.mapping.review_text_col == "Review Text"
    assert result.mapping.rating_col == "Rating"
    assert result.missing_required == []


# ===========================================================================
# Schema B — Lowercase column names
# ===========================================================================

def test_lowercase_columns_detected():
    """Test B: 'review text' and 'rating' (all lowercase) → compatible."""
    df = _review_df(review_col="review text", rating_col="rating")
    result = detect_schema(df)
    assert result.compatible is True
    assert result.mapping.review_text_col == "review text"
    assert result.mapping.rating_col == "rating"


def test_lowercase_no_space():
    """'reviewtext' and 'rating' → compatible."""
    df = _review_df(review_col="reviewtext", rating_col="rating")
    result = detect_schema(df)
    assert result.compatible is True


# ===========================================================================
# Schema C — Alias column names
# ===========================================================================

def test_customer_review_and_score_alias():
    """Test C: 'customer_review' and 'score' → compatible."""
    df = _review_df(review_col="customer_review", rating_col="score")
    result = detect_schema(df)
    assert result.compatible is True
    assert result.mapping.review_text_col == "customer_review"
    assert result.mapping.rating_col == "score"


def test_comment_and_stars_alias():
    """'comment' and 'stars' → compatible."""
    df = _review_df(review_col="comment", rating_col="stars")
    result = detect_schema(df)
    assert result.compatible is True


def test_feedback_and_overall_alias():
    """'feedback' and 'overall' → compatible."""
    df = _review_df(review_col="feedback", rating_col="overall")
    result = detect_schema(df)
    assert result.compatible is True


# ===========================================================================
# Schema D — Underscore / mixed case column names
# ===========================================================================

def test_underscore_review_text():
    """Test D: 'Review_Text' and 'Rating' → compatible via normalization."""
    df = _review_df(review_col="Review_Text", rating_col="Rating")
    result = detect_schema(df)
    assert result.compatible is True
    assert result.mapping.review_text_col == "Review_Text"


def test_mixed_case_aliases():
    """'Review_Content' and 'Review_Rating' → compatible."""
    df = _review_df(review_col="Review_Content", rating_col="Review_Rating")
    result = detect_schema(df)
    assert result.compatible is True


# ===========================================================================
# Schema E — Unsupported dataset
# ===========================================================================

def test_unsupported_dataset_rejected():
    """Test E: employee/sales dataset → incompatible."""
    df = pd.DataFrame({
        "Customer_ID": [1, 2, 3],
        "Salary": [50000, 60000, 55000],
        "Age": [30, 40, 35],
        "City": ["Mumbai", "Delhi", "Pune"],
        "Purchase_Amount": [1200.0, 3400.0, 800.0],
    })
    result = detect_schema(df)
    assert result.compatible is False
    assert "Review Text" in result.missing_required
    assert "Rating" in result.missing_required


def test_missing_review_text_only():
    """CSV with Rating but no Review Text → incompatible, error mentions Review Text."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3],
        "product": ["shoe", "hat", "bag"],
    })
    result = detect_schema(df)
    assert result.compatible is False
    assert "Review Text" in result.missing_required
    assert "Rating" not in result.missing_required


def test_missing_rating_only():
    """CSV with Review Text but no Rating → incompatible, error mentions Rating."""
    df = pd.DataFrame({
        "Review Text": ["Good product", "Bad fit", "Loved it"],
        "product": ["shoe", "hat", "bag"],
    })
    result = detect_schema(df)
    assert result.compatible is False
    assert "Rating" in result.missing_required
    assert "Review Text" not in result.missing_required


# ===========================================================================
# Optional column absence
# ===========================================================================

def test_optional_columns_absence_is_graceful():
    """Dataset with only Review Text + Rating → compatible, many optional missing."""
    df = _review_df()   # minimal — only Review Text and Rating
    result = detect_schema(df)
    assert result.compatible is True
    assert "Title" in result.optional_missing
    assert "Department Name" in result.optional_missing
    assert "Class Name" in result.optional_missing


def test_optional_department_detected_when_present():
    """If 'department' column present, it maps to 'Department Name'."""
    df = _review_df(extra={"department": ["Tops", "Tops", "Dresses", "Bottoms", "Tops"]})
    result = detect_schema(df)
    assert result.compatible is True
    assert "Department Name" in result.optional_found
    assert result.optional_found["Department Name"] == "department"


# ===========================================================================
# apply_mapping
# ===========================================================================

def test_apply_mapping_renames_correctly():
    """apply_mapping should rename aliased columns to canonical names."""
    df = _review_df(review_col="review", rating_col="stars",
                    extra={"title": ["T1", "T2", "T3", "T4", "T5"]})
    result = detect_schema(df)
    assert result.compatible is True
    mapped_df = apply_mapping(df, result)
    assert "Review Text" in mapped_df.columns
    assert "Rating" in mapped_df.columns


def test_apply_mapping_preserves_original_data():
    """After apply_mapping, the data values must be identical."""
    df = _review_df(review_col="review", rating_col="stars")
    result = detect_schema(df)
    mapped_df = apply_mapping(df, result)
    assert list(mapped_df["Review Text"]) == list(df["review"])
    assert list(mapped_df["Rating"]) == list(df["stars"])


# ===========================================================================
# Rating validation
# ===========================================================================

def test_numeric_rating_detected():
    """Standard 1–5 numeric rating → is_numeric True, min=1, max=5."""
    df = _review_df()
    result = detect_schema(df)
    assert result.rating_is_numeric == True  # noqa: E712 — tolerates np.True_
    assert result.rating_min == 1
    assert result.rating_max == 5


def test_critical_threshold_standard_scale():
    """Standard 1–5 scale → default threshold of 2, is_default True."""
    df = _review_df()
    result = detect_schema(df)
    threshold, is_default = get_critical_threshold(result)
    assert threshold == 2.0
    assert is_default is True


def test_critical_threshold_non_standard_scale_flagged():
    """Rating scale 1–10 → is_default False (user confirmation needed)."""
    df = pd.DataFrame({
        "Review Text": ["Good", "Bad", "Okay", "Terrible", "Average"],
        "Rating": [8, 2, 5, 1, 6],
    })
    result = detect_schema(df)
    assert result.compatible is True
    threshold, is_default = get_critical_threshold(result)
    assert is_default is False


def test_critical_threshold_2_to_5_scale_flagged():
    """Rating scale 2–5 → is_default False (user confirmation needed)."""
    df = pd.DataFrame({
        "Review Text": ["Bad", "Okay", "Good", "Great"],
        "Rating": [2, 3, 4, 5],
    })
    result = detect_schema(df)
    assert result.compatible is True
    threshold, is_default = get_critical_threshold(result)
    assert is_default is False


# ===========================================================================
# filter_critical_reviews with threshold
# ===========================================================================

def test_filter_critical_with_custom_threshold():
    """filter_critical_reviews should respect custom threshold parameter."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Review Text": ["r1", "r2", "r3", "r4", "r5"],
        "clean_text": ["r1", "r2", "r3", "r4", "r5"],
    })
    critical = filter_critical_reviews(df, threshold=3)
    assert set(critical["Rating"].unique()) == {1, 2, 3}
    assert len(critical) == 3


def test_filter_critical_default_threshold_unchanged():
    """Default threshold of 2 must still work (backward compatibility)."""
    df = pd.DataFrame({
        "Rating": [1, 2, 3, 4, 5],
        "Review Text": ["r1", "r2", "r3", "r4", "r5"],
        "clean_text": ["r1", "r2", "r3", "r4", "r5"],
    })
    critical = filter_critical_reviews(df)
    assert len(critical) == 2
    assert set(critical["Rating"].unique()) == {1, 2}


# ===========================================================================
# select_top_critical_reviews — dynamic min rating
# ===========================================================================

def test_select_top_uses_minimum_rating():
    """With minimum rating = 1, selection should pick Rating==1 rows."""
    df = pd.DataFrame({
        "Rating": [5, 1, 1, 2, 1, 1],
        "Title": ["G", "S1", "L1", "T2", "M1", "VS"],
        "Review Text": [
            "This was wonderful.",
            "Terrible shirt.",
            "This dress had terrible seams, awful fabric, completely itchy and see through.",
            "Did not fit well.",
            "Awful material, buttons fell off immediately upon opening.",
            "Bad.",
        ],
        "clean_text": ["c1", "c2", "c3", "c4", "c5", "c6"],
    })
    top_3 = select_top_critical_reviews(df, n=3)
    assert top_3.count == 3
    assert all(r.rating == 1 for r in top_3.reviews)
    # Descending length order
    lengths = [r.review_length for r in top_3.reviews]
    assert lengths == sorted(lengths, reverse=True)


def test_select_top_with_non_one_minimum():
    """Dataset where minimum is 2 → select Rating==2 rows."""
    df = pd.DataFrame({
        "Rating": [5, 2, 2, 3, 2, 4],
        "Review Text": [
            "Perfect item.",
            "Very poor quality.",
            "The sizing was completely off and material was cheap and see through.",
            "Average experience.",
            "Returned it, bad quality.",
            "Pretty good.",
        ],
        "clean_text": ["c1", "c2", "c3", "c4", "c5", "c6"],
    })
    top_3 = select_top_critical_reviews(df, n=3)
    assert top_3.count == 3
    assert all(r.rating == 2 for r in top_3.reviews)


def test_custom_dataset_pipeline_with_2_to_5_ratings():
    """End-to-end test on custom dataset containing:
    review_content, rating, review_title, category, product_id
    with ratings [2, 3, 4, 5].
    Verifies:
    - schema detection works
    - mapping works
    - threshold filtering works with threshold=2 and threshold=3
    - data cleaning and overview work without TypeError
    """
    from src.cleaning import clean_dataset
    from src.analysis import get_dataset_overview, filter_critical_reviews

    custom_df = pd.DataFrame({
        "review_content": [
            "Terrible fabric, ripped after one wear.",
            "Average item, runs a bit small.",
            "Good quality overall.",
            "Absolutely loved it, fits perfectly!"
        ],
        "rating": [2, 3, 4, 5],
        "review_title": ["Awful", "Okay", "Nice", "Love"],
        "category": ["Dresses", "Tops", "Bottoms", "Dresses"],
        "product_id": [101, 102, 103, 104],
    })

    # 1. Schema detection
    detection = detect_schema(custom_df)
    assert detection.compatible is True
    assert detection.review_text_col == "review_content"
    assert detection.rating_col == "rating"
    assert detection.rating_min == 2
    assert detection.rating_max == 5

    # 2. Mapping
    mapped_df = apply_mapping(custom_df, detection)
    assert "Review Text" in mapped_df.columns
    assert "Rating" in mapped_df.columns
    assert "Title" in mapped_df.columns  # mapped from review_title
    assert "Department Name" in mapped_df.columns  # mapped from category
    assert "Clothing ID" in mapped_df.columns  # mapped from product_id

    # 3. Cleaning
    clean_df, metrics = clean_dataset(mapped_df)
    assert len(clean_df) == 4

    # 4. Threshold filtering
    crit_2 = filter_critical_reviews(clean_df, threshold=2)
    assert len(crit_2) == 1
    assert crit_2.iloc[0]["Rating"] == 2

    crit_3 = filter_critical_reviews(clean_df, threshold=3)
    assert len(crit_3) == 2
    assert set(crit_3["Rating"].tolist()) == {2, 3}

    # 5. Dataset overview with custom threshold
    overview_2 = get_dataset_overview(mapped_df, clean_df, threshold=2)
    assert overview_2.critical_reviews_count == 1

    overview_3 = get_dataset_overview(mapped_df, clean_df, threshold=3)
    assert overview_3.critical_reviews_count == 2

