"""Customer Feedback Intelligence & AI Response System — Streamlit App

This project analyzes customer reviews using Python and Pandas,
identifies critical negative reviews using rating-based rules,
analyzes common complaint keywords, and uses Google Gemini to generate
personalized apology-response drafts.

Dataset columns used:
    reviewText  — customer review
    overall     — rating (1–5)
    summary     — review title
    asin        — product ID
    helpful     — helpful vote/count
"""

import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import DATA_PATH, GEMINI_MODEL, REQUIRED_COLUMNS
from src.cleaning import load_dataset, clean_dataset, normalize_columns, validate_columns
from src.analysis import filter_critical_reviews, get_dataset_overview, detect_complaints, query_reviews
from src.complaint_analysis import get_keyword_insights
from src.review_selection import select_top_3_reviews
from src.gemini_service import generate_apology_email
from src.validators import validate_response

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Feedback Intelligence & AI Response System",
    page_icon="🛍️",
    layout="wide",
)

# ── Custom styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .review-quote {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        font-style: italic;
        color: #1e293b;
        margin-bottom: 8px;
    }
    .badge-pass {
        background: #dcfce7; color: #166534;
        padding: 3px 10px; border-radius: 6px;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
    }
    .badge-fail {
        background: #fee2e2; color: #991b1b;
        padding: 3px 10px; border-radius: 6px;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached data loading ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(file_source):
    """Load CSV, normalize columns, validate, and clean the dataset."""
    try:
        if isinstance(file_source, Path):
            df = pd.read_csv(file_source)
        else:
            df = pd.read_csv(file_source)

        df = normalize_columns(df)
        ok, missing = validate_columns(df)
        if not ok:
            return None, None, None, (
                f"Missing required columns: {', '.join(missing)}\n\n"
                f"Required: {REQUIRED_COLUMNS}"
            )

        df_clean, cleaning_summary = clean_dataset(df)
        overview = get_dataset_overview(df, df_clean)
        return df, df_clean, overview, None

    except Exception as exc:
        return None, None, None, str(exc)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🛍️ Feedback System")
st.sidebar.markdown("---")

# Data source
source_option = st.sidebar.radio("Data Source", ["Default Dataset", "Upload CSV"])

file_to_load = None
if source_option == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV", type=["csv"],
        help=f"Required columns: {', '.join(REQUIRED_COLUMNS)}"
    )
    if uploaded:
        file_to_load = uploaded
    else:
        st.sidebar.info("Upload a CSV file to begin.")
else:
    if DATA_PATH.exists():
        file_to_load = DATA_PATH
    else:
        st.sidebar.warning("Default dataset not found. Please upload the CSV.")
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="default_upload")
        if uploaded:
            file_to_load = uploaded

# Welcome screen if no file loaded
if file_to_load is None:
    st.title("🛍️ Customer Feedback Intelligence & AI Response System")
    st.markdown("""
    This project analyzes customer reviews using Python and Pandas,
    identifies critical negative reviews using rating-based rules,
    analyzes common complaint keywords, and uses Google Gemini to generate
    personalized apology-response drafts.

    **Required dataset columns:**
    - `reviewText` — customer review
    - `overall` — rating (1–5)
    - `summary` — review title
    - `asin` — product ID
    - `helpful` — helpful vote/count

    Use the sidebar to load the dataset.
    """)
    st.stop()

# Load data
with st.spinner("Loading dataset..."):
    df_raw, df_clean, overview, error = load_and_clean(file_to_load)

if error:
    st.error(error)
    st.stop()

critical_df = filter_critical_reviews(df_clean)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.metric("Raw Rows", f"{overview['total_raw_rows']:,}")
st.sidebar.metric("Cleaned Rows", f"{overview['cleaned_rows']:,}")
st.sidebar.metric("Critical Reviews", f"{overview['critical_count']:,} ({overview['critical_pct']}%)")
st.sidebar.markdown("**Critical Rule:** `overall <= 2`")
st.sidebar.markdown("---")

# Navigation (6 pages)
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🔍 Complaint Analysis",
    "🚨 Critical Reviews",
    "🤖 AI Response Generator",
    "✅ Validation",
    "ℹ️ Methodology",
])

# API key status
has_key = bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_key_here")
if has_key:
    st.sidebar.success("✅ Gemini API key set")
else:
    st.sidebar.warning("⚠️ No API key — fallback mode")


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.title("📊 Dataset Overview")
    st.markdown(
        "This project analyzes customer reviews using Python and Pandas, "
        "identifies critical negative reviews using rating-based rules, "
        "analyzes common complaint keywords, and uses Google Gemini to generate "
        "personalized apology-response drafts."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Rows", f"{overview['total_raw_rows']:,}")
    col2.metric("Cleaned Rows", f"{overview['cleaned_rows']:,}")
    col3.metric("Critical (≤ 2 ★)", f"{overview['critical_count']:,}")
    col4.metric("Critical Rate", f"{overview['critical_pct']}%")

    st.markdown("---")
    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("Rating Distribution")
        rating_df = pd.DataFrame([
            {"Rating": f"{k} ★", "Count": v}
            for k, v in sorted(overview["rating_distribution"].items())
        ])
        colors = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"]
        fig = px.bar(rating_df, x="Rating", y="Count", text="Count",
                     color="Rating", color_discrete_sequence=colors)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        st.subheader("Top Complaint Terms (Critical Reviews)")
        insights = get_keyword_insights(critical_df)
        term_df = pd.DataFrame(insights["predefined_terms"][:8])
        fig2 = px.bar(term_df, x="count", y="term", orientation="h",
                      color="count", color_continuous_scale="Reds", text="count")
        fig2.update_layout(
            yaxis={"autorange": "reversed"}, height=340,
            coloraxis_showscale=False, margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🧹 Cleaning Details"):
        st.markdown(f"""
        | Step | Value |
        |------|-------|
        | Rows before cleaning | {overview['total_raw_rows']:,} |
        | Rows after cleaning | {overview['cleaned_rows']:,} |
        | Rows removed | {overview['total_raw_rows'] - overview['cleaned_rows']:,} |

        **Steps applied:**
        - Converted `overall` to numeric, dropped invalid values
        - Dropped rows with missing `reviewText` or `overall`
        - Filled missing `summary` with empty string
        - Kept only valid ratings between 1 and 5
        - Removed exact duplicate rows
        - Created `clean_text` column (lowercase, letters only) for keyword analysis
        - Preserved original `reviewText`
        """)


# ============================================================
# PAGE 2: COMPLAINT ANALYSIS
# ============================================================
elif page == "🔍 Complaint Analysis":
    st.title("🔍 Complaint Keyword Analysis")
    st.markdown("What are customers complaining about in low-rated reviews (`overall <= 2`)?")

    if critical_df.empty:
        st.warning("No critical reviews found in this dataset.")
        st.stop()

    insights = get_keyword_insights(critical_df)

    tab1, tab2, tab3 = st.tabs([
        "Predefined Complaint Terms",
        "Most Frequent Words",
        "Common 2-Word Phrases (Bigrams)",
    ])

    with tab1:
        st.markdown("**How often does each tracked complaint term appear?**")
        term_df = pd.DataFrame(insights["predefined_terms"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(term_df.rename(columns={"term": "Term", "count": "Count"}),
                         use_container_width=True, hide_index=True)
        with col_b:
            fig = px.bar(term_df, x="term", y="count", text="count",
                         color="count", color_continuous_scale="Reds")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, height=380, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("**Most frequent meaningful words in critical reviews:**")
        kw_df = pd.DataFrame(insights["top_keywords"])
        fig = px.bar(kw_df, x="term", y="count", text="count",
                      color="count", color_continuous_scale="Blues")
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=420, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("**Most common 2-word phrases in critical reviews:**")
        bg_df = pd.DataFrame(insights["top_bigrams"])
        fig = px.bar(bg_df, x="count", y="phrase", orientation="h",
                     color="count", color_continuous_scale="Viridis", text="count")
        fig.update_layout(yaxis={"autorange": "reversed"}, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3: CRITICAL REVIEWS
# ============================================================
elif page == "🚨 Critical Reviews":
    st.title("🚨 Critical Reviews")
    st.markdown(f"Critical review rule: `overall <= 2` (**{len(critical_df):,}** total reviews).")

    tab_top, tab_queue = st.tabs(["⭐ Top Critical Reviews", "📋 All Critical Reviews Queue"])

    with tab_top:
        st.subheader("Top Critical Reviews")
        st.markdown("""
        **Selection Rule:**
        1. Filter for `overall == 1` (1-star reviews).
        2. Sort by review text length (longest first).
        
        Longer reviews provide specific complaints that give Gemini the context needed to draft a personalized response.
        """)

        try:
            top3 = select_top_3_reviews(df_clean)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        for i, review in enumerate(top3):
            st.markdown(f"### #{i+1} — {review['summary'] or '(No Summary)'}")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Rating:** {'★' * review['overall']}{'☆' * (5 - review['overall'])}")
            c2.markdown(f"**Length:** {review['review_length']} chars")
            c3.markdown(f"**Product ID (asin):** `{review['asin']}`")
            c4.markdown(f"**Helpful Votes:** {review['helpful']}")

            st.markdown(
                f'<div class="review-quote">"{review["reviewText"]}"</div>',
                unsafe_allow_html=True
            )

            if review["detected_complaints"]:
                badges = " ".join([f"`{c}`" for c in review["detected_complaints"]])
                st.markdown(f"**Complaint keywords detected:** {badges}")

            if st.button(f"Load into AI Response Generator →", key=f"load_{i}"):
                st.session_state["ai_review_text"] = review["reviewText"]
                st.session_state["ai_rating"] = review["overall"]
                st.session_state["ai_summary"] = review["summary"]
                st.session_state["ai_asin"] = review["asin"]
                st.info("Loaded! Navigate to '🤖 AI Response Generator' in the sidebar.")

            st.markdown("---")

    with tab_queue:
        st.subheader("All Critical Reviews Queue")
        st.markdown(f"Filter and inspect all **{len(critical_df):,}** critical reviews (`overall <= 2`).")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            rating_choice = st.selectbox("Filter by rating", ["All (1 & 2 ★)", "1 ★ only", "2 ★ only"])
        with col2:
            all_asins = ["All"] + sorted(df_clean["asin"].dropna().astype(str).unique().tolist()[:50])
            asin_choice = st.selectbox("Filter by Product (asin)", all_asins)
        with col3:
            search = st.text_input("Search keyword", placeholder="e.g., fabric, return, size...")

        subset = critical_df.copy()
        if rating_choice == "1 ★ only":
            subset = subset[subset["overall"] == 1]
        elif rating_choice == "2 ★ only":
            subset = subset[subset["overall"] == 2]
        if asin_choice != "All":
            subset = subset[subset["asin"].astype(str) == asin_choice]
        if search.strip():
            kw = search.strip().lower()
            mask = subset["clean_text"].str.contains(kw, na=False)
            mask = mask | subset["summary"].astype(str).str.lower().str.contains(kw, na=False)
            subset = subset[mask]

        st.caption(f"Showing **{len(subset):,}** reviews")
        display_cols = ["overall", "summary", "reviewText", "asin", "helpful"]
        st.dataframe(subset[display_cols], use_container_width=True, height=420)


# ============================================================
# PAGE 4: AI RESPONSE GENERATOR
# ============================================================
elif page == "🤖 AI Response Generator":
    st.title("🤖 AI Response Generator")
    st.markdown("""
    Generate personalized apology-response drafts for critical reviews using Google Gemini.

    > ⚠️ **Human Review:** AI response drafts must always be reviewed by a team member before sending.
    """)

    # Pre-fill options from top critical reviews
    try:
        top3 = select_top_3_reviews(df_clean)
        presets = {
            f"Case {i+1}: {r['summary'][:40] or 'Review'} (asin: {r['asin']})": r
            for i, r in enumerate(top3)
        }
    except Exception:
        presets = {}

    preset_choice = st.selectbox(
        "Quick-load a top critical review:",
        ["(Enter manually below)"] + list(presets.keys())
    )

    if preset_choice != "(Enter manually below)" and preset_choice in presets:
        chosen = presets[preset_choice]
        default_text = chosen["reviewText"]
        default_rating = chosen["overall"]
        default_summary = chosen["summary"]
        default_asin = chosen["asin"]
    else:
        default_text = st.session_state.get("ai_review_text", "")
        default_rating = st.session_state.get("ai_rating", 1)
        default_summary = st.session_state.get("ai_summary", "")
        default_asin = st.session_state.get("ai_asin", "")

    col_text, col_meta = st.columns([3, 1])
    with col_text:
        review_input = st.text_area(
            "Customer Review Text (reviewText) *",
            value=default_text,
            height=150,
            placeholder="Paste the customer review text here..."
        )
    with col_meta:
        rating_input = st.number_input("Rating (overall: 1–5)", min_value=1, max_value=5, value=int(default_rating))
        summary_input = st.text_input("Review Title (summary)", value=default_summary)
        asin_input = st.text_input("Product ID (asin)", value=default_asin)

    if st.button("🚀 Generate AI Apology Email", type="primary", use_container_width=True):
        if not review_input.strip():
            st.error("Please enter some review text first.")
        else:
            with st.spinner("Calling Google Gemini..."):
                draft = generate_apology_email(
                    review_text=review_input,
                    rating=rating_input,
                    summary=summary_input,
                    asin=asin_input,
                )
            st.session_state["draft"] = draft
            st.session_state["draft_review"] = review_input
            st.session_state["draft_rating"] = rating_input

    if "draft" in st.session_state:
        draft = st.session_state["draft"]
        st.markdown("---")
        st.subheader("Generated Draft")

        if draft["is_fallback"]:
            st.warning("⚠️ Fallback template used — Gemini API returned an error or API key is not configured.")
            if draft["error"]:
                with st.expander("Error details"):
                    st.code(draft["error"])

        st.markdown(f"**Subject:** `{draft['subject']}`")
        st.text_area("Email Body (editable):", value=draft["email_body"], height=200, key="editable_body")

        # Validation badges
        v = draft["validation"]
        st.subheader("Quality Validation Checks")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            badge = "badge-pass" if v["word_limit_ok"] else "badge-fail"
            icon = "✅" if v["word_limit_ok"] else "❌"
            st.markdown(f'<span class="{badge}">{icon} {v["word_count"]}/130 words</span>', unsafe_allow_html=True)
        with c2:
            badge = "badge-pass" if v["has_signoff"] else "badge-fail"
            icon = "✅" if v["has_signoff"] else "❌"
            st.markdown(f'<span class="{badge}">{icon} Sign-off</span>', unsafe_allow_html=True)
        with c3:
            badge = "badge-pass" if v["has_next_step"] else "badge-fail"
            icon = "✅" if v["has_next_step"] else "❌"
            st.markdown(f'<span class="{badge}">{icon} Next step offered</span>', unsafe_allow_html=True)
        with c4:
            badge = "badge-pass" if v["no_fabrication"] else "badge-fail"
            icon = "✅" if v["no_fabrication"] else "❌"
            st.markdown(f'<span class="{badge}">{icon} No false claims</span>', unsafe_allow_html=True)

        st.markdown("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ Approve Draft", use_container_width=True):
                st.success("Draft approved! Ready for human dispatch.")
        with b2:
            if st.button("🔄 Regenerate", use_container_width=True):
                del st.session_state["draft"]
                st.rerun()


# ============================================================
# PAGE 5: VALIDATION
# ============================================================
elif page == "✅ Validation":
    st.title("✅ Response Validation")
    st.markdown("""
    Test any apology email against the defined quality and safety rules:
    - **Word count ≤ 130**
    - **Customer Care Team** sign-off
    - **Concrete next step** offered (refund, return, or exchange)
    - **No fabricated actions** (no claiming actions have already been completed, no fake order numbers)
    - **Personalized** (references specific words from customer complaint)
    """)

    sample_email = (
        "Dear Customer,\n\n"
        "We are truly sorry that the dress was completely see-through and did not fit as expected. "
        "We understand how disappointing this must be and would be glad to help arrange a full refund or exchange. "
        "Please reply to this email so we can assist you right away.\n\n"
        "Sincerely,\nCustomer Care Team"
    )

    email_input = st.text_area("Draft Email to Validate:", value=sample_email, height=180)
    review_input = st.text_input("Original Review Text (for specificity check):",
                                 value="Fabric was completely see-through and the fit was awful.")
    rating_input = st.slider("Original Rating:", 1, 5, 1)

    if st.button("Run Validation Checks", type="primary"):
        result = validate_response(email_input, review_input, rating_input)

        if result["passed_all"]:
            st.success("🎉 All checks passed! This email meets all quality guidelines.")
        else:
            st.error("⚠️ One or more checks failed. Review the details below.")

        checks = {
            "Not Empty": result["not_empty"],
            f"Word Count ({result['word_count']}/130)": result["word_limit_ok"],
            "Has 'Customer Care Team' Sign-off": result["has_signoff"],
            "Offers Next Step (refund/return/exchange)": result["has_next_step"],
            "No False Completed-Action Claims": result["no_fabrication"],
            "References Specific Complaint": result["is_specific"],
        }

        for check, passed in checks.items():
            icon = "✅" if passed else "❌"
            st.markdown(f"{icon} **{check}**")

        if result["overlapping_keywords"]:
            st.caption(f"Matching keywords: {', '.join(result['overlapping_keywords'])}")
        if result["fabricated_phrases"]:
            st.error(f"Problematic phrases found: {result['fabricated_phrases']}")


# ============================================================
# PAGE 6: METHODOLOGY
# ============================================================
elif page == "ℹ️ Methodology":
    st.title("ℹ️ Project Methodology")
    st.markdown("""
    ## Customer Feedback Intelligence & AI Response System

    This project analyzes customer reviews using Python and Pandas,
    identifies critical negative reviews using rating-based rules,
    analyzes common complaint keywords, and uses Google Gemini to generate
    personalized apology-response drafts.

    ---

    ### Dataset Columns

    | Column | Meaning | Description |
    |--------|---------|-------------|
    | `reviewText` | Customer review | The full text of the customer's review |
    | `overall` | Rating | Star rating given by the customer (1 to 5) |
    | `summary` | Review title | Short headline summarizing the review |
    | `asin` | Product ID | Unique product identifier |
    | `helpful` | Helpful vote/count | Number of customers who found the review helpful |

    ---

    ### Project Workflow

    ```
    CSV
     ↓
    Pandas Cleaning
     ↓
    Critical Reviews (overall <= 2)
     ↓
    Complaint Keyword Analysis
     ↓
    Top Critical Reviews (overall == 1, longest reviewText)
     ↓
    Gemini AI Response
     ↓
    Validation
     ↓
    Human Review
    ```

    ---

    ### Safety & Response Guidelines

    Every generated AI response adheres to the following rules:
    - **Under 130 words** to remain concise and respectful of customer time.
    - **Empathetic & Personalized**: Acknowledges customer frustration and quotes/addresses their specific issues.
    - **Concrete Next Step**: Offers a refund, return, or exchange upon reply.
    - **Sign-off**: Signed off specifically as `Customer Care Team`.
    - **No Fabrication**: Does not invent order numbers, refund amounts, or company policies.
    - **No Pre-Claimed Actions**: Does not claim an action has already been processed or completed.
    - **Human Review**: Drafts must be approved by a human team member before being sent out.

    ---

    ### Tools Used

    - **Python** (Pandas, collections.Counter, re)
    - **Streamlit** (interactive web UI)
    - **Plotly** (visualizations)
    - **Google Gemini API** (apology email draft generation)
    - **python-dotenv** (API configuration)
    - **pytest** (unit testing)
    """)
