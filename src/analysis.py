"""Statistical and rule-based review analysis module.

Works with both the official Imarticus dataset and any compatible dataset whose
columns have been normalized to canonical names via :mod:`src.schema_detection`.
All optional column accesses are guarded against KeyError / AttributeError.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd

from src.schemas import DatasetOverview, ReviewItem
from src.config import PREDEFINED_COMPLAINT_TERMS


def filter_critical_reviews(df: pd.DataFrame, threshold: Union[int, float] = 2) -> pd.DataFrame:
    """Filter reviews using the mandatory rule-based condition: Rating <= threshold.

    The default threshold of 2 implements the official assessment rule (Rating <= 2).
    For non-standard rating scales or custom datasets, callers may supply a threshold.

    Validates:
    - 'Rating' column exists in df (raises ValueError with descriptive message if missing)
    - Threshold is numeric (raises ValueError if non-numeric)
    - 'Rating' values can be interpreted numerically (raises ValueError if entirely non-numeric)
    - Original DataFrame is preserved without in-place modification
    - Returns a new DataFrame copy containing only matching rows

    NO machine learning model or trained classifier is utilized.
    This rule is fully auditable and deterministic.
    """
    if "Rating" not in df.columns:
        raise ValueError(
            "Column 'Rating' is missing from DataFrame. "
            "Please ensure the dataset contains a mapped 'Rating' column before filtering."
        )

    try:
        num_threshold = float(threshold)
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid threshold '{threshold}'. Threshold must be a numeric value."
        )

    if df.empty:
        return df.copy()

    # Safely interpret Rating as numeric without mutating input DataFrame
    numeric_ratings = pd.to_numeric(df["Rating"], errors="coerce")
    if numeric_ratings.dropna().empty:
        raise ValueError(
            "Column 'Rating' contains non-numeric values that cannot be interpreted as ratings."
        )

    # Boolean condition: Rating <= num_threshold (ignoring NaNs)
    mask = (numeric_ratings <= num_threshold) & numeric_ratings.notna()
    return df[mask].copy()


def detect_complaint_indicators(text: str) -> List[str]:
    """Detect presence of predefined complaint terms in review text."""
    lower_text = str(text).lower()
    matched = []
    for term in PREDEFINED_COMPLAINT_TERMS:
        if term == "see-through":
            if "see-through" in lower_text or "see through" in lower_text:
                matched.append(term)
        else:
            if term in lower_text:
                matched.append(term)
    return matched


def _safe_col(row: pd.Series, col: str, cast=str, default=None):
    """Safely retrieve a column value; return default if column missing or NaN."""
    val = row.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        return default


def get_dataset_overview(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    threshold: Union[int, float] = 2
) -> DatasetOverview:
    """Calculate aggregate dataset metrics and distributions."""
    critical_df = filter_critical_reviews(df_cleaned, threshold=threshold)
    total_cleaned = len(df_cleaned)
    critical_count = len(critical_df)
    critical_pct = round((critical_count / total_cleaned * 100), 2) if total_cleaned > 0 else 0.0

    rating_counts = df_cleaned["Rating"].value_counts().to_dict()
    rating_dist = {int(k): int(v) for k, v in rating_counts.items()}

    # Department breakdown (only if column exists)
    if "Department Name" in df_cleaned.columns:
        dept_series = df_cleaned["Department Name"].fillna("Unspecified")
        dept_dist = {str(k): int(v) for k, v in dept_series.value_counts().head(8).to_dict().items()}
    else:
        dept_dist = {"All": total_cleaned}

    # Calculate top complaint term in critical reviews
    from src.complaint_analysis import get_predefined_term_counts
    term_counts = get_predefined_term_counts(critical_df)
    top_complaint = term_counts[0].term if term_counts else "fit"

    return DatasetOverview(
        total_raw_rows=len(df_raw),
        cleaned_rows=total_cleaned,
        critical_reviews_count=critical_count,
        critical_percentage=critical_pct,
        one_star_count=int(rating_dist.get(1, 0)),
        two_star_count=int(rating_dist.get(2, 0)),
        rating_distribution=rating_dist,
        department_distribution=dept_dist,
        top_complaint_term=top_complaint,
    )


def query_reviews(
    df: pd.DataFrame,
    rating: Optional[int] = None,
    is_critical_only: bool = False,
    department: Optional[str] = None,
    search_keyword: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    critical_threshold: Union[int, float] = 2,
) -> List[ReviewItem]:
    """Filter and page reviews for UI display."""
    subset = df.copy()

    if is_critical_only:
        subset = filter_critical_reviews(subset, threshold=critical_threshold)
    elif rating is not None:
        subset = subset[subset["Rating"] == rating]

    if department and department.strip() and "Department Name" in subset.columns:
        subset = subset[subset["Department Name"].astype(str).str.lower() == department.strip().lower()]

    if search_keyword and search_keyword.strip():
        term = search_keyword.strip().lower()
        mask = subset["clean_text"].str.contains(term, na=False, regex=False)
        if "Title" in subset.columns:
            mask = mask | subset["Title"].astype(str).str.lower().str.contains(term, na=False)
        subset = subset[mask]

    # Slice requested window
    paged = subset.iloc[offset: offset + limit]

    items: List[ReviewItem] = []
    for idx, row in paged.iterrows():
        review_text = str(row.get("Review Text", ""))
        rating_val = int(row.get("Rating", 0))
        items.append(
            ReviewItem(
                index=int(idx),
                clothing_id=_safe_col(row, "Clothing ID", int),
                age=_safe_col(row, "Age", int),
                title=_safe_col(row, "Title", str, default=""),
                review_text=review_text,
                rating=rating_val,
                recommended=_safe_col(row, "Recommended IND", int),
                positive_feedback_count=_safe_col(row, "Positive Feedback Count", int),
                division_name=_safe_col(row, "Division Name", str),
                department_name=_safe_col(row, "Department Name", str),
                class_name=_safe_col(row, "Class Name", str),
                clean_text=str(row.get("clean_text", "")),
                review_length=len(review_text),
                is_critical=rating_val <= critical_threshold,
                detected_complaints=detect_complaint_indicators(review_text),
            )
        )
    return items
