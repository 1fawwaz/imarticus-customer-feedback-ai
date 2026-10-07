"""Deterministic critical review selection module.

Works with both the official Imarticus dataset (Rating == 1) and any compatible
dataset (minimum rating in the cleaned DataFrame).
"""

from typing import List
import pandas as pd

from src.schemas import Top3ReviewsResponse, ReviewItem
from src.analysis import detect_complaint_indicators, _safe_col


def select_top_critical_reviews(df: pd.DataFrame, n: int = 3) -> Top3ReviewsResponse:
    """Select the most critical detailed reviews using deterministic logic.

    Selection Rule (official assessment):
      1. Filter for the lowest rating in the dataset (Rating == min_rating).
      2. Rank descending by character length of the original 'Review Text'.
      3. Select top n reviews.

    For the official Women's Clothing dataset the minimum rating is always 1,
    so the behaviour is identical to the original implementation.

    Rationale:
      Lowest-rated reviews represent the strongest customer dissatisfaction.
      Longer review texts contain detailed feedback on specific product defects,
      fit discrepancies, and customer experience, providing essential context
      for a personalized and empathetic apology response.
    """
    min_rating = int(df["Rating"].min())
    min_star_df = df[df["Rating"] == min_rating].copy()

    if len(min_star_df) < n:
        raise ValueError(
            f"Dataset contains only {len(min_star_df)} {min_rating}-star reviews, "
            f"which is fewer than the requested {n} reviews."
        )

    # Sort descending by original character length
    min_star_df["review_length"] = min_star_df["Review Text"].astype(str).str.len()
    top_df = min_star_df.sort_values(by="review_length", ascending=False).head(n)

    selection_rule = (
        f"Rating == {min_rating} (minimum in dataset), "
        "sorted descending by original Review Text character length"
    )

    reviews: List[ReviewItem] = []
    for idx, row in top_df.iterrows():
        review_text = str(row["Review Text"])
        reviews.append(
            ReviewItem(
                index=int(idx),
                clothing_id=_safe_col(row, "Clothing ID", int),
                age=_safe_col(row, "Age", int),
                title=_safe_col(row, "Title", str, default=""),
                review_text=review_text,
                rating=int(row["Rating"]),
                recommended=_safe_col(row, "Recommended IND", int),
                positive_feedback_count=_safe_col(row, "Positive Feedback Count", int),
                division_name=_safe_col(row, "Division Name", str),
                department_name=_safe_col(row, "Department Name", str),
                class_name=_safe_col(row, "Class Name", str),
                clean_text=str(row.get("clean_text", "")),
                review_length=int(row["review_length"]),
                is_critical=True,
                detected_complaints=detect_complaint_indicators(review_text),
            )
        )

    return Top3ReviewsResponse(
        selection_rule=selection_rule,
        count=len(reviews),
        reviews=reviews,
    )
