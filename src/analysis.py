"""Review analysis functions — filtering, overview stats, and querying."""

from typing import List, Optional
import pandas as pd

from src.config import COMPLAINT_TERMS


def filter_critical_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Return only critical reviews where rating (overall) is 1 or 2.
    
    This is the core rule defined in the assessment:
        Critical review = overall <= 2
    """
    return df[df["overall"] <= 2].copy()


def get_dataset_overview(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """Calculate basic stats about the dataset after cleaning.
    
    Returns a dict with:
    - raw and cleaned row counts
    - critical review count and percentage
    - rating distribution (1 to 5 stars)
    - top complaint term found in critical reviews
    """
    critical = filter_critical_reviews(df_clean)
    total = len(df_clean)
    critical_count = len(critical)
    critical_pct = round((critical_count / total * 100), 2) if total > 0 else 0.0

    rating_dist = {
        int(k): int(v)
        for k, v in df_clean["overall"].value_counts().to_dict().items()
    }

    # Find most common predefined complaint term
    from src.complaint_analysis import get_predefined_term_counts
    term_counts = get_predefined_term_counts(critical)
    top_term = term_counts[0]["term"] if term_counts else "fit"

    return {
        "total_raw_rows": len(df_raw),
        "cleaned_rows": total,
        "critical_count": critical_count,
        "critical_pct": critical_pct,
        "one_star_count": int(rating_dist.get(1, 0)),
        "two_star_count": int(rating_dist.get(2, 0)),
        "rating_distribution": rating_dist,
        "top_complaint_term": top_term,
    }


def detect_complaints(text: str) -> List[str]:
    """Check which predefined complaint words appear in a review."""
    lower = str(text).lower()
    found = []
    for term in COMPLAINT_TERMS:
        if term == "see-through":
            if "see-through" in lower or "see through" in lower:
                found.append(term)
        elif term in lower:
            found.append(term)
    return found


def query_reviews(
    df: pd.DataFrame,
    rating: Optional[int] = None,
    critical_only: bool = False,
    keyword: Optional[str] = None,
    asin: Optional[str] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Filter the reviews DataFrame for display in the app.
    
    You can filter by:
    - critical_only: show only rating <= 2 reviews
    - rating: show only a specific star rating
    - keyword: search in the review text or summary
    - asin: filter by product ID
    """
    subset = df.copy()

    if critical_only:
        subset = filter_critical_reviews(subset)
    elif rating is not None:
        subset = subset[subset["overall"] == rating]

    if asin and asin.strip():
        subset = subset[subset["asin"].astype(str).str.lower() == asin.strip().lower()]

    if keyword and keyword.strip():
        kw = keyword.strip().lower()
        mask = subset["clean_text"].str.contains(kw, na=False)
        mask = mask | subset["summary"].astype(str).str.lower().str.contains(kw, na=False)
        subset = subset[mask]

    return subset.head(limit)
