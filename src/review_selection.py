"""Select the top 3 most critical reviews from the dataset."""

from typing import List
import pandas as pd

from src.analysis import detect_complaints


def select_top_3_reviews(df: pd.DataFrame) -> List[dict]:
    """Pick the 3 most useful reviews to draft AI responses for.

    Selection rule:
    1. Keep only the lowest-rated reviews (e.g., all 1-star reviews).
    2. Sort by review text length (longest first) — longer reviews contain
       more specific complaints, which leads to better AI responses.
    3. Return the top 3.

    Returns a list of 3 dicts, each with all review fields plus detected complaints.
    """
    if "overall" not in df.columns or "reviewText" not in df.columns:
        raise ValueError("DataFrame must have 'overall' and 'reviewText' columns.")

    min_rating = int(df["overall"].min())
    lowest_rated = df[df["overall"] == min_rating].copy()

    if len(lowest_rated) < 3:
        raise ValueError(
            f"Not enough {min_rating}-star reviews (found {len(lowest_rated)}, need at least 3)."
        )

    lowest_rated["review_length"] = lowest_rated["reviewText"].astype(str).str.len()
    top3 = lowest_rated.sort_values("review_length", ascending=False).head(3)

    results = []
    for idx, row in top3.iterrows():
        review_text = str(row["reviewText"])
        results.append({
            "index": int(idx),
            "asin": str(row.get("asin", "")),
            "overall": int(row["overall"]),
            "summary": str(row.get("summary", "")),
            "reviewText": review_text,
            "helpful": row.get("helpful", 0),
            "review_length": int(row["review_length"]),
            "detected_complaints": detect_complaints(review_text),
            "selection_rule": f"overall == {min_rating}, sorted by review length (descending)",
        })

    return results
