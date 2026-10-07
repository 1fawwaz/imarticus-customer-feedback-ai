"""Batch pipeline script — runs the full analysis and saves results to outputs/.

Run this directly:
    python -m src.automation

Or from the project root:
    python src/automation.py
"""

from pathlib import Path
import pandas as pd

from src.config import DATA_PATH, OUTPUTS_DIR
from src.cleaning import load_dataset, clean_dataset
from src.analysis import filter_critical_reviews, get_dataset_overview
from src.complaint_analysis import get_keyword_insights
from src.review_selection import select_top_3_reviews
from src.gemini_service import generate_apology_email


def run_pipeline(csv_path: Path = DATA_PATH, output_dir: Path = OUTPUTS_DIR):
    """Run the full analysis pipeline and save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading dataset from: {csv_path}")
    df_raw = load_dataset(csv_path)

    print("[2/5] Cleaning data...")
    df_clean, cleaning_summary = clean_dataset(df_raw)
    print(f"      Rows before: {cleaning_summary['rows_before']:,}")
    print(f"      Rows after:  {cleaning_summary['rows_after']:,}")
    print(f"      Removed:     {cleaning_summary['rows_removed']:,}")

    print("[3/5] Filtering critical reviews (overall <= 2)...")
    critical_df = filter_critical_reviews(df_clean)
    print(f"      Critical reviews: {len(critical_df):,}")

    print("[4/5] Analyzing complaint keywords...")
    insights = get_keyword_insights(critical_df)

    print("[5/5] Selecting top 3 reviews and generating AI responses...")
    top3 = select_top_3_reviews(df_clean)

    drafts = []
    for review in top3:
        draft = generate_apology_email(
            review_text=review["reviewText"],
            rating=review["overall"],
            summary=review["summary"],
            asin=review["asin"],
        )
        drafts.append({
            "Rating": review["overall"],
            "Summary": review["summary"],
            "ASIN": review["asin"],
            "Review Length": review["review_length"],
            "Original Review": review["reviewText"],
            "Email Subject": draft["subject"],
            "AI Apology Body": draft["email_body"],
            "Model Used": draft["model_used"],
            "Is Fallback": draft["is_fallback"],
            "Word Count": draft["validation"]["word_count"],
            "Passed All Checks": draft["validation"]["passed_all"],
        })

    # Save outputs
    critical_path = output_dir / "critical_reviews.csv"
    complaint_path = output_dir / "complaint_summary.csv"
    drafts_path = output_dir / "ai_response_drafts.csv"

    critical_df.to_csv(critical_path, index=False)
    pd.DataFrame(insights["predefined_terms"]).to_csv(complaint_path, index=False)
    pd.DataFrame(drafts).to_csv(drafts_path, index=False)

    print("\nDone! Results saved to outputs/:")
    print(f"  • {critical_path.name}")
    print(f"  • {complaint_path.name}")
    print(f"  • {drafts_path.name}")


if __name__ == "__main__":
    run_pipeline()
