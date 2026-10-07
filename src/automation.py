"""End-to-end batch processing pipeline for customer feedback intelligence.

Executes the end-to-end assessment and decision-support workflow:
  1. Ingest raw CSV data.
  2. Clean data using deterministic rules.
  3. Filter critical feedback (Rating <= 2).
  4. Compute complaint keyword, term, and bigram distributions.
  5. Deterministically select the top 3 most critical detailed reviews (Rating == 1, longest text).
  6. Generate Gemini apology email drafts with strict safety and quality criteria.
  7. Validate drafts against word limit, anti-fabrication, and required sign-off.
  8. Save artifacts to outputs/ directory for human-in-the-loop review.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import DATA_PATH, OUTPUTS_DIR
from src.cleaning import load_raw_dataset, clean_dataset
from src.analysis import filter_critical_reviews, get_dataset_overview
from src.complaint_analysis import get_keyword_insights, get_predefined_term_counts
from src.review_selection import select_top_critical_reviews
from src.gemini_service import generate_apology_email


def run_batch_pipeline(csv_path: Path = DATA_PATH, output_dir: Path = OUTPUTS_DIR) -> Dict[str, Any]:
    """Execute the batch analytics and response generation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading raw dataset from: {csv_path}")
    df_raw = load_raw_dataset(csv_path)

    print("[2/6] Cleaning dataset...")
    df_clean, metrics = clean_dataset(df_raw)

    print("[3/6] Applying rule-based critical review filtering (Rating <= 2)...")
    critical_df = filter_critical_reviews(df_clean)

    print("[4/6] Analyzing complaint keywords and bigrams...")
    insights = get_keyword_insights(critical_df)

    print("[5/6] Selecting top 3 detailed 1-star reviews...")
    top_3_response = select_top_critical_reviews(df_clean, n=3)

    print("[6/6] Generating and validating AI apology email drafts...")
    drafts_records = []
    for item in top_3_response.reviews:
        draft = generate_apology_email(
            review_text=item.review_text,
            rating=item.rating,
            title=item.title,
            clothing_id=item.clothing_id,
            department=item.department_name,
        )
        drafts_records.append({
            "Review Index": item.index,
            "Rating": item.rating,
            "Title": item.title,
            "Review Length": item.review_length,
            "Original Review": item.review_text,
            "Email Subject": draft.subject,
            "AI Apology Body": draft.email_body,
            "Model Used": draft.model_used,
            "Is Fallback": draft.is_fallback,
            "Word Count": draft.validation.word_count,
            "Passed All Validations": draft.validation.passed_all,
            "Validation Signoff": draft.validation.signoff,
            "Validation Word Limit": draft.validation.word_limit,
            "Validation Next Step": draft.validation.next_step,
            "Validation No Fabricated Action": draft.validation.no_fabricated_action,
            "Validation Specificity": draft.validation.specificity,
            "Human Review Status": "PENDING_APPROVAL",
        })

    # Save artifacts to outputs/
    critical_output_path = output_dir / "critical_reviews.csv"
    complaint_output_path = output_dir / "complaint_summary.csv"
    drafts_output_path = output_dir / "ai_response_drafts.csv"

    critical_df.to_csv(critical_output_path, index=False)
    
    complaint_summary_df = pd.DataFrame([
        {"Complaint Term": item.term, "Frequency": item.count}
        for item in insights.predefined_terms
    ])
    complaint_summary_df.to_csv(complaint_output_path, index=False)

    drafts_df = pd.DataFrame(drafts_records)
    drafts_df.to_csv(drafts_output_path, index=False)

    print("\n--- Pipeline Completed Successfully ---")
    print(f"  • Critical Reviews Saved: {critical_output_path}")
    print(f"  • Complaint Summary Saved: {complaint_output_path}")
    print(f"  • AI Response Drafts Saved: {drafts_output_path}")

    return {
        "status": "SUCCESS",
        "cleaned_rows": len(df_clean),
        "critical_reviews": len(critical_df),
        "top_3_selected": len(top_3_response.reviews),
        "drafts_generated": len(drafts_records),
    }


if __name__ == "__main__":
    run_batch_pipeline()
