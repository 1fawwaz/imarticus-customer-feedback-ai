"""Smart CSV schema detection and column mapping for the Customer Feedback Intelligence system.

Provides automatic detection of review-dataset columns regardless of exact naming,
casing, or spacing.  Falls back to a user-facing mapping UI when detection is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Column alias registry
# ---------------------------------------------------------------------------

#: Aliases that map to the canonical "Review Text" field (required)
REVIEW_TEXT_ALIASES: List[str] = [
    "review text",
    "reviewtext",
    "review_text",
    "review",
    "customer review",
    "customer_review",
    "customerreview",
    "review content",
    "review_content",
    "reviewcontent",
    "comment",
    "comments",
    "feedback",
    "product review",
    "product_review",
    "text",
    "body",
    "description",
]

#: Aliases that map to the canonical "Rating" field (required)
RATING_ALIASES: List[str] = [
    "rating",
    "stars",
    "star",
    "score",
    "review rating",
    "review_rating",
    "reviewrating",
    "customer rating",
    "customer_rating",
    "user rating",
    "user_rating",
    "overall",
    "overall rating",
    "overall_rating",
    "satisfaction",
    "satisfaction score",
]

#: Aliases for optional fields: {canonical_name: [aliases]}
OPTIONAL_ALIASES: Dict[str, List[str]] = {
    "Title": [
        "title",
        "review title",
        "review_title",
        "headline",
        "subject",
        "summary",
    ],
    "Department Name": [
        "department name",
        "department_name",
        "department",
        "dept",
        "dept name",
        "category",
        "product category",
        "product_category",
    ],
    "Class Name": [
        "class name",
        "class_name",
        "class",
        "sub category",
        "sub_category",
        "subcategory",
        "type",
        "product type",
        "product_type",
    ],
    "Clothing ID": [
        "clothing id",
        "clothing_id",
        "clothingid",
        "product id",
        "product_id",
        "productid",
        "item id",
        "item_id",
        "sku",
        "asin",
    ],
    "Age": [
        "age",
        "customer age",
        "reviewer age",
    ],
    "Division Name": [
        "division name",
        "division_name",
        "division",
    ],
    "Recommended IND": [
        "recommended ind",
        "recommended_ind",
        "recommended",
        "recommend",
        "would recommend",
    ],
    "Positive Feedback Count": [
        "positive feedback count",
        "positive_feedback_count",
        "helpful",
        "helpful votes",
        "helpful_votes",
        "thumbs up",
        "upvotes",
    ],
}

# ---------------------------------------------------------------------------
# Official Imarticus dataset fingerprint columns
# ---------------------------------------------------------------------------
OFFICIAL_SCHEMA_COLS: List[str] = [
    "Clothing ID",
    "Age",
    "Title",
    "Review Text",
    "Rating",
    "Recommended IND",
    "Positive Feedback Count",
    "Division Name",
    "Department Name",
    "Class Name",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ColumnMapping:
    """Maps canonical application column names to actual DataFrame column names."""
    # Required
    review_text_col: str          # actual column name for "Review Text"
    rating_col: str               # actual column name for "Rating"
    # Optional
    title_col: Optional[str] = None
    department_col: Optional[str] = None
    class_col: Optional[str] = None
    clothing_id_col: Optional[str] = None
    age_col: Optional[str] = None
    division_col: Optional[str] = None
    recommended_col: Optional[str] = None
    helpful_col: Optional[str] = None

    def to_rename_dict(self) -> Dict[str, str]:
        """Return {original_col: canonical_col} for renaming operations."""
        mapping: Dict[str, str] = {
            self.review_text_col: "Review Text",
            self.rating_col: "Rating",
        }
        optionals = {
            self.title_col: "Title",
            self.department_col: "Department Name",
            self.class_col: "Class Name",
            self.clothing_id_col: "Clothing ID",
            self.age_col: "Age",
            self.division_col: "Division Name",
            self.recommended_col: "Recommended IND",
            self.helpful_col: "Positive Feedback Count",
        }
        for original, canonical in optionals.items():
            if original and original != canonical:
                mapping[original] = canonical
        return mapping


@dataclass
class DetectionResult:
    """Complete result of CSV schema detection."""
    compatible: bool
    is_official_dataset: bool
    mapping: Optional[ColumnMapping]
    detected_columns: List[str]
    missing_required: List[str]
    optional_found: Dict[str, str]    # {canonical: actual}
    optional_missing: List[str]
    rating_min: Optional[float] = None
    rating_max: Optional[float] = None
    rating_is_numeric: bool = True
    needs_user_mapping: bool = False   # True when auto-detection was ambiguous
    ambiguous_fields: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _normalize(col: str) -> str:
    """Normalize a column name: lowercase, strip BOM, collapse whitespace."""
    # Remove BOM
    col = col.lstrip("\ufeff")
    # Strip surrounding whitespace and lower
    col = col.strip().lower()
    # Collapse repeated internal whitespace / underscores
    col = re.sub(r"[\s_]+", " ", col)
    return col


def _best_match(normalized_col: str, aliases: List[str]) -> bool:
    """Return True if normalized_col exactly matches any alias."""
    return normalized_col in aliases


def _find_col(df_cols: List[str], aliases: List[str]) -> Tuple[Optional[str], bool]:
    """Find the first DataFrame column matching any alias.

    Returns:
        (matched_original_col, is_confident)
        is_confident = True if exactly one match; False if multiple (ambiguous).
    """
    matches = [c for c in df_cols if _best_match(_normalize(c), aliases)]
    if len(matches) == 1:
        return matches[0], True
    if len(matches) > 1:
        return matches[0], False   # ambiguous — use first but flag it
    return None, True              # no match found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_schema(df: pd.DataFrame) -> DetectionResult:
    """Auto-detect the schema of an uploaded customer-review CSV.

    Returns a :class:`DetectionResult` describing what was found, whether the
    dataset is compatible, and the :class:`ColumnMapping` to use.
    """
    df_cols = list(df.columns)
    detected_cols = df_cols[:]
    ambiguous: List[str] = []

    # --- Required fields ---------------------------------------------------
    review_col, rv_conf = _find_col(df_cols, REVIEW_TEXT_ALIASES)
    rating_col, rt_conf = _find_col(df_cols, RATING_ALIASES)

    missing_required: List[str] = []
    if not review_col:
        missing_required.append("Review Text")
    if not rating_col:
        missing_required.append("Rating")

    if not rv_conf:
        ambiguous.append("Review Text")
    if not rt_conf:
        ambiguous.append("Rating")

    if missing_required:
        return DetectionResult(
            compatible=False,
            is_official_dataset=False,
            mapping=None,
            detected_columns=detected_cols,
            missing_required=missing_required,
            optional_found={},
            optional_missing=list(OPTIONAL_ALIASES.keys()),
            error=f"Required columns not found: {', '.join(missing_required)}",
        )

    # --- Optional fields ---------------------------------------------------
    optional_found: Dict[str, str] = {}
    optional_missing: List[str] = []

    title_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Title"])
    dept_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Department Name"])
    class_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Class Name"])
    clothing_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Clothing ID"])
    age_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Age"])
    div_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Division Name"])
    rec_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Recommended IND"])
    helpful_col, _ = _find_col(df_cols, OPTIONAL_ALIASES["Positive Feedback Count"])

    _opt_map = {
        "Title": title_col,
        "Department Name": dept_col,
        "Class Name": class_col,
        "Clothing ID": clothing_col,
        "Age": age_col,
        "Division Name": div_col,
        "Recommended IND": rec_col,
        "Positive Feedback Count": helpful_col,
    }
    for canon, actual in _opt_map.items():
        if actual:
            optional_found[canon] = actual
        else:
            optional_missing.append(canon)

    mapping = ColumnMapping(
        review_text_col=review_col,
        rating_col=rating_col,
        title_col=title_col,
        department_col=dept_col,
        class_col=class_col,
        clothing_id_col=clothing_col,
        age_col=age_col,
        division_col=div_col,
        recommended_col=rec_col,
        helpful_col=helpful_col,
    )

    # --- Rating analysis ---------------------------------------------------
    rating_series = df[rating_col]
    rating_numeric = pd.to_numeric(rating_series, errors="coerce")
    is_numeric = rating_numeric.notna().mean() > 0.9   # >90% parseable as number

    rating_min = float(rating_numeric.min()) if is_numeric else None
    rating_max = float(rating_numeric.max()) if is_numeric else None

    # --- Official dataset detection ----------------------------------------
    normalized_df_cols = {_normalize(c) for c in df_cols}
    normalized_official = {_normalize(c) for c in OFFICIAL_SCHEMA_COLS}
    is_official = normalized_official.issubset(normalized_df_cols)

    return DetectionResult(
        compatible=True,
        is_official_dataset=is_official,
        mapping=mapping,
        detected_columns=detected_cols,
        missing_required=[],
        optional_found=optional_found,
        optional_missing=optional_missing,
        rating_min=rating_min,
        rating_max=rating_max,
        rating_is_numeric=is_numeric,
        needs_user_mapping=bool(ambiguous),
        ambiguous_fields=ambiguous,
    )


def apply_mapping(df: pd.DataFrame, result: DetectionResult) -> pd.DataFrame:
    """Return a copy of df with columns renamed to canonical application names.

    Only renames columns that differ from their canonical names.  Preserves all
    other columns unchanged so no data is lost.
    """
    if result.mapping is None:
        raise ValueError("Cannot apply mapping: DetectionResult has no mapping.")
    rename_dict = result.mapping.to_rename_dict()
    # Only rename keys that actually exist in the DataFrame
    safe_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
    return df.rename(columns=safe_rename)


def get_critical_threshold(result: DetectionResult) -> Tuple[float, bool]:
    """Return (threshold, is_default) for critical review filtering.

    For official 1-5 scale datasets the assessment threshold of 2 is returned.
    For non-standard scales the caller should prompt the user.
    """
    if result.is_official_dataset:
        return 2.0, True
    if result.rating_min is not None and result.rating_max is not None:
        if result.rating_min >= 1 and result.rating_max <= 5:
            return 2.0, True   # Same scale, use default
    return 2.0, False           # Different scale — flag for user confirmation
