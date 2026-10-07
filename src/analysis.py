"""Statistical and rule-based review analysis module."""

from typing import Dict, Any, List, Optional
import pandas as pd

from src.schemas import DatasetOverview, ReviewItem
from src.config import PREDEFINED_COMPLAINT_TERMS


def filter_critical_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Filter reviews using the mandatory rule-based condition: Rating <= 2.

    NO machine learning model or trained classifier is utilized.
    This rule is fully auditable and deterministic.
    """
    return df[df["Rating"] <= 2].copy()


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


def get_dataset_overview(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> DatasetOverview:
    """Calculate aggregate dataset metrics and distributions."""
    critical_df = filter_critical_reviews(df_cleaned)
    total_cleaned = len(df_cleaned)
    critical_count = len(critical_df)
    critical_pct = round((critical_count / total_cleaned * 100), 2) if total_cleaned > 0 else 0.0

    rating_counts = df_cleaned["Rating"].value_counts().to_dict()
    rating_dist = {int(k): int(v) for k, v in rating_counts.items()}

    # Department breakdown (impute missing with 'Unknown')
    dept_series = df_cleaned["Department Name"].fillna("Unspecified")
    dept_dist = {str(k): int(v) for k, v in dept_series.value_counts().head(8).to_dict().items()}

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
) -> List[ReviewItem]:
    """Filter and page reviews for UI display."""
    subset = df.copy()

    if is_critical_only:
        subset = subset[subset["Rating"] <= 2]
    elif rating is not None:
        subset = subset[subset["Rating"] == rating]

    if department and department.strip():
        subset = subset[subset["Department Name"].astype(str).str.lower() == department.strip().lower()]

    if search_keyword and search_keyword.strip():
        term = search_keyword.strip().lower()
        subset = subset[subset["clean_text"].str.contains(term, na=False, regex=False)]

    # Slice requested window
    paged = subset.iloc[offset : offset + limit]

    items: List[ReviewItem] = []
    for idx, row in paged.iterrows():
        review_text = str(row.get("Review Text", ""))
        rating_val = int(row.get("Rating", 0))
        items.append(
            ReviewItem(
                index=int(idx),
                clothing_id=int(row["Clothing ID"]) if pd.notna(row.get("Clothing ID")) else None,
                age=int(row["Age"]) if pd.notna(row.get("Age")) else None,
                title=str(row.get("Title", "")),
                review_text=review_text,
                rating=rating_val,
                recommended=int(row["Recommended IND"]) if pd.notna(row.get("Recommended IND")) else None,
                positive_feedback_count=int(row["Positive Feedback Count"]) if pd.notna(row.get("Positive Feedback Count")) else None,
                division_name=str(row.get("Division Name", "")) if pd.notna(row.get("Division Name")) else None,
                department_name=str(row.get("Department Name", "")) if pd.notna(row.get("Department Name")) else None,
                class_name=str(row.get("Class Name", "")) if pd.notna(row.get("Class Name")) else None,
                clean_text=str(row.get("clean_text", "")),
                review_length=len(review_text),
                is_critical=rating_val <= 2,
                detected_complaints=detect_complaint_indicators(review_text),
            )
        )
    return items
