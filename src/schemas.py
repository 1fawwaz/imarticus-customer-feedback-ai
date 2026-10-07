"""Data schemas and transfer models for the feedback intelligence system."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CleaningMetrics(BaseModel):
    rows_before: int
    rows_after: int
    rows_removed: int
    missing_review_text_dropped: int
    missing_ratings_dropped: int
    duplicates_removed: int


class DatasetOverview(BaseModel):
    total_raw_rows: int
    cleaned_rows: int
    critical_reviews_count: int
    critical_percentage: float
    one_star_count: int
    two_star_count: int
    rating_distribution: Dict[int, int]
    department_distribution: Dict[str, int]
    top_complaint_term: str


class ComplaintTermFrequency(BaseModel):
    term: str
    count: int


class BigramFrequency(BaseModel):
    phrase: str
    count: int


class KeywordInsights(BaseModel):
    top_keywords: List[ComplaintTermFrequency]
    predefined_terms: List[ComplaintTermFrequency]
    top_bigrams: List[BigramFrequency]


class ReviewItem(BaseModel):
    index: int
    clothing_id: Optional[int] = None
    age: Optional[int] = None
    title: str
    review_text: str
    rating: int
    recommended: Optional[int] = None
    positive_feedback_count: Optional[int] = None
    division_name: Optional[str] = None
    department_name: Optional[str] = None
    class_name: Optional[str] = None
    clean_text: Optional[str] = None
    review_length: int
    is_critical: bool
    detected_complaints: List[str] = Field(default_factory=list)


class Top3ReviewsResponse(BaseModel):
    selection_rule: str
    count: int
    reviews: List[ReviewItem]


class ValidationResult(BaseModel):
    word_limit: bool = Field(description="Word count <= 130")
    word_count: int
    signoff: bool = Field(description="Contains 'Customer Care Team'")
    specificity: bool = Field(description="References specific complaints from the review")
    next_step: bool = Field(description="Offers concrete next step (refund, return, exchange)")
    no_fabricated_action: bool = Field(description="Does not falsely claim actions already occurred")
    not_empty: bool = Field(description="Response is non-empty")
    passed_all: bool
    details: Dict[str, str] = Field(default_factory=dict)


class AIResponseDraft(BaseModel):
    subject: str
    email_body: str
    model_used: str
    is_fallback: bool
    error_message: Optional[str] = None
    validation: ValidationResult


class GenerateResponseRequest(BaseModel):
    review_text: str
    rating: int
    title: Optional[str] = ""
    clothing_id: Optional[int] = None
    department: Optional[str] = None


class ValidateResponseRequest(BaseModel):
    email_text: str
    original_review_text: str
    rating: int


class FilterReviewsRequest(BaseModel):
    rating: Optional[int] = None
    is_critical_only: Optional[bool] = False
    department: Optional[str] = None
    search_keyword: Optional[str] = None
    limit: Optional[int] = 50
    offset: Optional[int] = 0
