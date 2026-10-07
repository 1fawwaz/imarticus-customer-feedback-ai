"""Deterministic critical review selection module."""

from typing import List
import pandas as pd

from src.schemas import Top3ReviewsResponse, ReviewItem
from src.analysis import detect_complaint_indicators


def select_top_critical_reviews(df: pd.DataFrame, n: int = 3) -> Top3ReviewsResponse:
    """Select the most critical detailed reviews using deterministic logic.

    Selection Rule:
      1. Filter for the lowest rating: Rating == 1.
      2. Rank descending by character length of the original 'Review Text'.
      3. Select top n reviews.

    Rationale:
      1-star reviews represent the strongest customer dissatisfaction.
      Longer review texts contain detailed feedback on specific product defects,
      fit discrepancies, and customer experience, providing the essential context
      needed for a personalized and empathetic apology response.
    """
    one_star_df = df[df["Rating"] == 1].copy()

    if len(one_star_df) < n:
        raise ValueError(
            f"Dataset contains only {len(one_star_df)} 1-star reviews, "
            f"which is fewer than the requested {n} reviews."
        )

    # Sort descending by original character length
    one_star_df["review_length"] = one_star_df["Review Text"].astype(str).str.len()
    top_df = one_star_df.sort_values(by="review_length", ascending=False).head(n)

    reviews: List[ReviewItem] = []
    for idx, row in top_df.iterrows():
        review_text = str(row["Review Text"])
        reviews.append(
            ReviewItem(
                index=int(idx),
                clothing_id=int(row["Clothing ID"]) if pd.notna(row.get("Clothing ID")) else None,
                age=int(row["Age"]) if pd.notna(row.get("Age")) else None,
                title=str(row.get("Title", "")),
                review_text=review_text,
                rating=int(row["Rating"]),
                recommended=int(row["Recommended IND"]) if pd.notna(row.get("Recommended IND")) else None,
                positive_feedback_count=int(row["Positive Feedback Count"]) if pd.notna(row.get("Positive Feedback Count")) else None,
                division_name=str(row.get("Division Name", "")) if pd.notna(row.get("Division Name")) else None,
                department_name=str(row.get("Department Name", "")) if pd.notna(row.get("Department Name")) else None,
                class_name=str(row.get("Class Name", "")) if pd.notna(row.get("Class Name")) else None,
                clean_text=str(row.get("clean_text", "")),
                review_length=int(row["review_length"]),
                is_critical=True,
                detected_complaints=detect_complaint_indicators(review_text),
            )
        )

    return Top3ReviewsResponse(
        selection_rule="Rating == 1, sorted descending by original Review Text character length",
        count=len(reviews),
        reviews=reviews,
    )
