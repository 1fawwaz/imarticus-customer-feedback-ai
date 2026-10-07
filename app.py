"""Customer Feedback Intelligence & AI Response System
Streamlit Production Application

An AI-powered customer complaint analysis and response assistant built for
the Imarticus Data Science Internship Assessment.

Supports the official Women's Clothing Reviews dataset AND any compatible
customer-review CSV via smart schema detection.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

from src.config import (
    DATA_PATH,
    GEMINI_MODEL,
    PREDEFINED_COMPLAINT_TERMS,
    SYSTEM_PROMPT,
)
from src.cleaning import clean_dataset, clean_review_text
from src.analysis import filter_critical_reviews, get_dataset_overview, detect_complaint_indicators, query_reviews
from src.complaint_analysis import (
    get_top_complaint_keywords,
    get_top_bigrams,
    get_predefined_term_counts,
    get_keyword_insights,
)
from src.review_selection import select_top_critical_reviews
from src.gemini_service import generate_apology_email
from src.validators import validate_apology_response
from src.schema_detection import detect_schema, apply_mapping, get_critical_threshold

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .badge-pass {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-fail {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .review-quote {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        font-style: italic;
        color: #1e293b;
    }
    .compat-check { font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Schema Detection & Mapping UI helper
# -----------------------------------------------------------------------------

def run_schema_detection_ui(df_raw: pd.DataFrame) -> tuple[pd.DataFrame | None, dict | None, str | None]:
    """Detect schema, show compatibility card, optional mapping UI.

    Returns:
        (mapped_df, detection_info_dict, error_message)
        error_message is None on success.
    """
    detection = detect_schema(df_raw)

    # --- Unsupported dataset ---
    if not detection.compatible:
        col_list = "\n".join(f"• {c}" for c in detection.detected_columns[:20])
        st.error(f"""
❌ **Unsupported Dataset**

This application analyzes customer-review datasets.

**Required fields not found:**
• Review Text
• Rating

**Detected columns in your file:**
```
{col_list}
```

Please upload a compatible customer-review CSV containing at least a **Review Text** and **Rating** column.
        """)
        return None, None, "Unsupported dataset."

    # --- Official dataset badge ---
    if detection.is_official_dataset:
        st.success("🟢 **Imarticus Assessment Dataset Detected** — Official Women's Clothing E-Commerce Reviews")
    else:
        st.info("🔵 **Compatible Customer Review Dataset** — Columns mapped automatically")

    # --- Rating scale check ---
    threshold, is_default = get_critical_threshold(detection)

    # --- Manual mapping if ambiguous ---
    if detection.needs_user_mapping:
        st.warning(f"⚠️ Automatic detection was ambiguous for: **{', '.join(detection.ambiguous_fields)}**. Please confirm below.")
        col_options = ["(None)"] + list(df_raw.columns)

        st.markdown("#### 🗺️ Map Your Dataset Columns")
        with st.form("column_mapping_form"):
            rt_choice = st.selectbox(
                "Review Text column *",
                options=[c for c in df_raw.columns],
                index=0,
                key="map_review_text"
            )
            rat_choice = st.selectbox(
                "Rating column *",
                options=[c for c in df_raw.columns],
                index=min(1, len(df_raw.columns) - 1),
                key="map_rating"
            )
            submitted = st.form_submit_button("✅ Confirm Dataset")

        if submitted:
            if rt_choice and rat_choice and rt_choice != rat_choice:
                rename_map = {rt_choice: "Review Text", rat_choice: "Rating"}
                df_mapped = df_raw.rename(columns=rename_map)
                st.success(f"Mapped → `Review Text`: `{rt_choice}` | `Rating`: `{rat_choice}`")
            else:
                st.error("Please select distinct columns for Review Text and Rating.")
                return None, None, "Mapping incomplete."
        else:
            return None, None, "Awaiting column mapping confirmation."

    else:
        # Automatic mapping (confident)
        df_mapped = apply_mapping(df_raw, detection)

    # --- Compatibility summary card ---
    with st.expander("📋 Dataset Compatibility Report", expanded=not detection.is_official_dataset):
        req_col, opt_col = st.columns(2)
        with req_col:
            st.markdown("**Required Fields**")
            st.markdown(f"✅ `Review Text` → `{detection.mapping.review_text_col}`")
            st.markdown(f"✅ `Rating` → `{detection.mapping.rating_col}`")
        with opt_col:
            st.markdown("**Optional Fields**")
            for canon, actual in detection.optional_found.items():
                st.markdown(f"✅ `{canon}` → `{actual}`")
            for missing in detection.optional_missing:
                st.markdown(f"❌ `{missing}` — not available")

        r_min = detection.rating_min
        r_max = detection.rating_max
        scale_label = f"{int(r_min)}–{int(r_max)}" if r_min is not None else "unknown"
        st.markdown(f"**Rating Scale Detected:** `{scale_label}`")

        if not detection.rating_is_numeric:
            st.error("⚠️ Rating column is not numeric. Please convert to numbers before uploading.")
            return None, None, "Non-numeric rating column."

    # --- Non-standard scale confirmation ---
    if not is_default:
        r_min = detection.rating_min or 1
        r_max = detection.rating_max or 10
        st.warning(f"⚠️ Rating scale `{int(r_min)}–{int(r_max)}` differs from the standard 1–5. Please confirm the critical-review threshold.")
        custom_threshold = st.slider(
            "Critical Rating Threshold (reviews at or below this are flagged as critical)",
            min_value=int(r_min),
            max_value=int(r_max),
            value=int(round((r_max - r_min) * 0.3 + r_min)),   # default ~30th percentile
            step=1,
            key="custom_threshold_slider"
        )
        if st.button("Confirm Threshold & Continue", key="confirm_threshold"):
            st.session_state["critical_threshold"] = custom_threshold
        elif "critical_threshold" not in st.session_state:
            st.info("Please confirm the critical threshold to continue analysis.")
            return None, None, "Awaiting threshold confirmation."

    info = {
        "detection": detection,
        "threshold": st.session_state.get("critical_threshold", int(threshold)),
        "is_official": detection.is_official_dataset,
        "optional_found": detection.optional_found,
    }
    return df_mapped, info, None


# -----------------------------------------------------------------------------
# Caching Data Pipelines
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raw_csv(file_source) -> tuple[pd.DataFrame | None, str | None]:
    """Read the raw CSV from a path or uploaded file object."""
    try:
        if isinstance(file_source, Path):
            if not file_source.exists():
                return None, f"Dataset file not found at `{file_source}`."
            return pd.read_csv(file_source), None
        else:
            return pd.read_csv(file_source), None
    except Exception as exc:
        return None, f"Failed to read CSV: {exc}"


@st.cache_data(show_spinner=False)
def process_mapped_data(df_mapped: pd.DataFrame) -> tuple:
    """Run cleaning pipeline on a mapped DataFrame."""
    df_cleaned, metrics = clean_dataset(df_mapped)
    overview = get_dataset_overview(df_mapped, df_cleaned)
    return df_cleaned, overview, metrics


@st.cache_data(show_spinner=False)
def compute_insights(df_critical: pd.DataFrame):
    """Compute top keywords, bigrams, and complaint counts dynamically."""
    return get_keyword_insights(df_critical)


# -----------------------------------------------------------------------------
# Sidebar — Data Source Selection
# -----------------------------------------------------------------------------
st.sidebar.title("🛍️ Feedback Intelligence")
st.sidebar.caption("AI-Powered Customer Complaint Analysis & Response Assistant")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data Source",
    ["Official Imarticus Dataset", "Upload Custom CSV"],
    index=0,
    key="data_source_radio"
)

uploaded_file = None
source_to_load = None

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload any customer review CSV. Required columns: Review Text + Rating (any naming)."
    )
    if uploaded_file is not None:
        source_to_load = uploaded_file
    else:
        st.sidebar.info("Upload a CSV or switch to the official dataset.")
else:
    # Official dataset — prefer local file, otherwise ask for upload
    if DATA_PATH.exists():
        source_to_load = DATA_PATH
    else:
        st.sidebar.warning("Official dataset not found locally.")
        alt_upload = st.sidebar.file_uploader(
            "Upload the official dataset CSV",
            type=["csv"],
            key="official_upload",
            help="Download from Kaggle: Women's E-Commerce Clothing Reviews"
        )
        if alt_upload:
            source_to_load = alt_upload
        else:
            st.sidebar.info("Download `Womens Clothing E-Commerce Reviews.csv` from Kaggle and upload it above.")

# ─── Load raw CSV ────────────────────────────────────────────────────────────
if source_to_load is None:
    # Nothing selected yet — show welcome screen
    st.title("🛍️ Customer Feedback Intelligence")
    st.markdown("""
    ### Welcome

    This application analyses customer review datasets and generates
    AI-powered apology email drafts using Google Gemini.

    **To get started:**
    - Select **Official Imarticus Dataset** in the sidebar (or upload the CSV if not found locally), OR
    - Select **Upload Custom CSV** to use your own customer review data.

    **Compatible datasets** need at least these two columns (any naming):
    | Purpose | Example names |
    |---|---|
    | Review Text | `Review Text`, `review`, `comment`, `feedback`, `customer_review` |
    | Rating | `Rating`, `stars`, `score`, `overall`, `satisfaction` |
    """)
    st.stop()

df_raw, load_error = load_raw_csv(source_to_load)
if load_error:
    st.error(load_error)
    st.stop()

# ─── Schema Detection ─────────────────────────────────────────────────────────
# For official local path with exact column names, skip the expanded UI
_is_path_source = isinstance(source_to_load, Path)
_detection_quick = detect_schema(df_raw)

if _detection_quick.is_official_dataset and _is_path_source:
    # Fast path: official dataset loaded from local file — no UI needed
    df_mapped = df_raw.copy()
    dataset_info = {
        "detection": _detection_quick,
        "threshold": 2,
        "is_official": True,
        "optional_found": _detection_quick.optional_found,
    }
    mapping_error = None
else:
    # Full schema detection + UI (for uploaded files or non-official CSVs)
    with st.container():
        st.markdown("### 🔍 Dataset Detection")
        df_mapped, dataset_info, mapping_error = run_schema_detection_ui(df_raw)

    if mapping_error or df_mapped is None:
        st.stop()

# ─── Clean & process ─────────────────────────────────────────────────────────
try:
    df_cleaned, overview, cleaning_metrics = process_mapped_data(df_mapped)
except Exception as exc:
    st.error(f"Data processing failed: {exc}")
    st.stop()

_threshold = dataset_info.get("threshold", 2)
_is_official = dataset_info.get("is_official", False)


# ─── Sidebar: critical threshold & Gemini status ─────────────────────────────
st.sidebar.markdown("---")
if _is_official:
    st.sidebar.success("🟢 Imarticus Dataset")
else:
    st.sidebar.info("🔵 Compatible Dataset")

st.sidebar.caption(f"**Critical threshold:** Rating ≤ {_threshold}")
st.sidebar.caption(f"**Rows:** {overview.cleaned_rows:,} cleaned / {overview.total_raw_rows:,} raw")

# Navigation pages
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview Dashboard",
        "📊 Complaint Intelligence",
        "🚨 Critical Reviews Queue",
        "⭐ Top 3 Severe Complaints",
        "🤖 AI Response Generator",
        "✅ Safety & Validation",
        "ℹ️ About / Methodology"
    ],
    index=0
)

st.sidebar.markdown("---")
configured_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
st.sidebar.caption(f"**Gemini Model**: `{configured_model}`")
has_key = bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_key_here")
if has_key:
    st.sidebar.success("API Key: Configured")
else:
    st.sidebar.warning("API Key: Fallback Mode")

# Custom complaint terms (sidebar — optional power-user feature)
st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Custom Complaint Terms", expanded=False):
    st.caption("Enter additional terms to track (comma-separated).")
    custom_terms_raw = st.text_input(
        "Terms",
        value="",
        placeholder="e.g., damaged, wrinkled, faded",
        key="custom_terms_input"
    )
    custom_terms: list[str] = [
        t.strip().lower() for t in custom_terms_raw.split(",") if t.strip()
    ]


# -----------------------------------------------------------------------------
# Helper: critical DataFrame with configurable threshold
# -----------------------------------------------------------------------------
def get_critical_df(df: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
    return filter_critical_reviews(df, threshold=threshold)


# =============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# =============================================================================
if page == "🏠 Overview Dashboard":
    st.title("Customer Feedback Intelligence")
    if _is_official:
        st.subheader("🟢 Overview Dashboard — Imarticus Assessment Dataset")
    else:
        st.subheader("🔵 Overview Dashboard — Custom Dataset")
    st.markdown("Automated analysis and decision-support metrics for retail customer feedback.")

    critical_df = get_critical_df(df_cleaned, _threshold)

    # Recompute critical count / pct from actual threshold (may differ from overview)
    actual_critical_count = len(critical_df)
    actual_critical_pct = round(actual_critical_count / overview.cleaned_rows * 100, 2) if overview.cleaned_rows else 0.0
    min_rating = int(df_cleaned["Rating"].min()) if len(df_cleaned) else 1

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Raw Reviews</div>
            <div class="metric-value">{overview.total_raw_rows:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cleaned Reviews</div>
            <div class="metric-value">{overview.cleaned_rows:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical (≤ {_threshold}★)</div>
            <div class="metric-value">{actual_critical_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical Rate</div>
            <div class="metric-value">{actual_critical_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{min_rating}-Star (Severe)</div>
            <div class="metric-value">{overview.rating_distribution.get(min_rating, 0):,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Star Rating Distribution")
        rating_df = pd.DataFrame([
            {"Rating": f"{k} Stars", "Count": v, "Stars": k}
            for k, v in sorted(overview.rating_distribution.items())
        ])
        colors = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"]
        fig_rating = px.bar(
            rating_df,
            x="Rating",
            y="Count",
            color="Rating",
            color_discrete_sequence=colors,
            text="Count",
        )
        fig_rating.update_traces(textposition="outside")
        fig_rating.update_layout(showlegend=False, height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_rating, use_container_width=True)

    with col_chart2:
        dept_label = "Department Name" if "Department Name" in df_cleaned.columns else "Category"
        st.markdown(f"#### Reviews by {dept_label}")
        dept_df = pd.DataFrame([
            {"Department": k, "Count": v}
            for k, v in overview.department_distribution.items()
        ])
        fig_dept = px.pie(
            dept_df,
            names="Department",
            values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_dept.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_dept, use_container_width=True)

    with st.expander("🔍 View Data Cleaning Audit Details", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Ingested Rows", f"{overview.total_raw_rows:,}")
        c2.metric("Post-Cleaning Rows", f"{overview.cleaned_rows:,}")
        c3.metric("Artifacts Removed", f"{overview.total_raw_rows - overview.cleaned_rows:,}")

        det = dataset_info.get("detection")
        col_summary = []
        if det:
            col_summary.append(f"- **Review Text** mapped from `{det.mapping.review_text_col}`")
            col_summary.append(f"- **Rating** mapped from `{det.mapping.rating_col}`")
            if det.optional_found:
                for canon, actual in det.optional_found.items():
                    col_summary.append(f"- **{canon}** mapped from `{actual}`")
            if det.optional_missing:
                col_summary.append(f"- Optional columns not found: {', '.join(f'`{c}`' for c in det.optional_missing)}")

        st.markdown("\n".join(col_summary) if col_summary else "")
        st.markdown("""
        **Cleaning Rules Applied:**
        - Dropped `Unnamed: 0` index artifact if present.
        - Converted Rating to numeric (coercing invalids to NaN).
        - Whitespace-only review text treated as missing.
        - Dropped rows missing Review Text or Rating.
        - Imputed missing Title with empty string (if column available).
        - Filtered ratings to valid 1–5 range.
        - Removed exact duplicate entries.
        - Generated `clean_text` column for keyword indexing.
        """)

    # Custom complaint terms section (if user entered any)
    if custom_terms:
        st.markdown("---")
        st.markdown("#### 🔧 Custom Complaint Term Frequencies")
        from collections import Counter
        combined_text = " ".join(df_cleaned["clean_text"].dropna().tolist())
        custom_counts = {}
        for term in custom_terms:
            # Handle multi-word terms
            custom_counts[term] = combined_text.count(term.replace("-", " ").replace("_", " "))
        cust_df = pd.DataFrame(list(custom_counts.items()), columns=["Term", "Count"])
        cust_df = cust_df.sort_values("Count", ascending=False)
        st.dataframe(cust_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 2: COMPLAINT INTELLIGENCE
# =============================================================================
elif page == "📊 Complaint Intelligence":
    st.title("Complaint Intelligence")
    st.markdown("Deep dive into recurring complaint themes, keywords, and adjacent word pairs.")

    critical_df = get_critical_df(df_cleaned, _threshold)

    if critical_df.empty:
        st.warning(f"No critical reviews found (Rating ≤ {_threshold}). Try adjusting the threshold.")
        st.stop()

    # Compute insights
    insights = compute_insights(critical_df)

    tab_terms, tab_keywords, tab_bigrams, tab_insights = st.tabs([
        "Predefined Complaint Terms",
        "Top Frequent Complaint Words",
        "Common Bigrams (Word Pairs)",
        "Automated Business Insights"
    ])

    with tab_terms:
        st.markdown("#### Frequency of Predefined Problem Categories")
        st.caption(f"Calculated dynamically across all critical reviews (Rating ≤ {_threshold}).")

        terms_df = pd.DataFrame([
            {
                "Term": t.term,
                "Mentions": t.count,
                "Prevalence (% of Critical)": round((t.count / len(critical_df)) * 100, 2)
            }
            for t in insights.predefined_terms
        ])

        fig_terms = px.bar(
            terms_df,
            x="Term",
            y="Mentions",
            color="Mentions",
            color_continuous_scale="Reds",
            text="Mentions",
        )
        fig_terms.update_traces(textposition="outside")
        fig_terms.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_terms, use_container_width=True)
        st.dataframe(terms_df, use_container_width=True, hide_index=True)

        # Custom terms tab-addon
        if custom_terms:
            st.markdown("---")
            st.markdown("**Custom Complaint Terms (from sidebar):**")
            combined_critical_text = " ".join(critical_df["clean_text"].dropna().tolist())
            cust_rows = []
            for term in custom_terms:
                norm = term.replace("-", " ").replace("_", " ")
                cnt = combined_critical_text.count(norm)
                pct = round(cnt / len(critical_df) * 100, 2) if len(critical_df) else 0
                cust_rows.append({"Term": term, "Mentions": cnt, "Prevalence (% of Critical)": pct})
            st.dataframe(pd.DataFrame(cust_rows), use_container_width=True, hide_index=True)

    with tab_keywords:
        st.markdown("#### Top 15 Complaint Keywords")
        st.caption("Extracted using Python `collections.Counter` with custom stopword filtering.")

        kw_df = pd.DataFrame([
            {"Keyword": k.term, "Frequency": k.count}
            for k in insights.top_keywords
        ])
        fig_kw = px.bar(
            kw_df,
            x="Frequency",
            y="Keyword",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Blues",
            text="Frequency"
        )
        fig_kw.update_layout(yaxis=dict(autorange="reversed"), height=450, coloraxis_showscale=False)
        fig_kw.update_traces(textposition="outside")
        st.plotly_chart(fig_kw, use_container_width=True)

    with tab_bigrams:
        st.markdown("#### Top Adjacent Word Pairs (Bigrams)")
        st.caption("Reveals the most frequent multi-word complaint patterns.")

        bg_df = pd.DataFrame([
            {"Bigram Phrase": b.phrase, "Occurrences": b.count}
            for b in insights.top_bigrams
        ])
        fig_bg = px.bar(
            bg_df,
            x="Occurrences",
            y="Bigram Phrase",
            orientation="h",
            color="Occurrences",
            color_continuous_scale="Teal",
            text="Occurrences"
        )
        fig_bg.update_layout(yaxis=dict(autorange="reversed"), height=420, coloraxis_showscale=False)
        fig_bg.update_traces(textposition="outside")
        st.plotly_chart(fig_bg, use_container_width=True)

    with tab_insights:
        st.markdown("#### Data-Driven Executive Insights")
        top_complaint = insights.predefined_terms[0].term if insights.predefined_terms else "fit"
        top_count = insights.predefined_terms[0].count if insights.predefined_terms else 0
        second_complaint = insights.predefined_terms[1].term if len(insights.predefined_terms) > 1 else "fabric"
        second_count = insights.predefined_terms[1].count if len(insights.predefined_terms) > 1 else 0

        small_count = next((t.count for t in insights.predefined_terms if t.term == "small"), 0)
        large_count = next((t.count for t in insights.predefined_terms if t.term == "large"), 0)

        st.info(f"""
        **Key Findings from Dynamic Dataset Calculation:**
        - **Primary Complaint Driver:** `{top_complaint.upper()}` is the most prominent dissatisfaction factor with **{top_count:,} mentions** across critical reviews.
        - **Secondary Complaint Driver:** `{second_complaint.upper()}` follows with **{second_count:,} mentions**.
        - **Sizing Asymmetry:** Terms `small` ({small_count:,}) and `large` ({large_count:,}) indicate size inconsistency is a primary trigger for returns.
        - **Bigram Patterns:** Frequent phrases validate specific physical defects in product quality.
        """)

        st.markdown("##### Strategic Recommendations:")
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.markdown("""
            **For Merchandising & Sizing Teams:**
            1. Standardize fit charts with detailed measurement notes.
            2. Investigate high-return SKUs for sizing variance.
            """)
        with col_rec2:
            st.markdown("""
            **For Quality Control & Support Teams:**
            1. Proactive customer outreach for lowest-rated reviews.
            2. Streamlined exchange workflows for repeat complaint categories.
            """)


# =============================================================================
# PAGE 3: CRITICAL REVIEWS QUEUE
# =============================================================================
elif page == "🚨 Critical Reviews Queue":
    st.title("Critical Reviews Queue")
    st.markdown(f"Filter, search, and review all critical feedback identified by rule `Rating ≤ {_threshold}`.")

    critical_df = get_critical_df(df_cleaned, _threshold)

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])

    min_r = int(df_cleaned["Rating"].min()) if len(df_cleaned) else 1
    max_r = int(_threshold)

    with col_f1:
        rating_opts = ["All Critical"] + [f"{r} Star" for r in range(min_r, max_r + 1)]
        rating_filter = st.selectbox("Rating Filter", rating_opts)

    with col_f2:
        if "Department Name" in df_cleaned.columns:
            depts = ["All Departments"] + sorted([str(d) for d in df_cleaned["Department Name"].dropna().unique()])
            dept_filter = st.selectbox("Department", depts)
        else:
            dept_filter = "All Departments"
            st.caption("(Department column not available)")

    with col_f3:
        search_query = st.text_input("🔍 Search keyword in review or title", placeholder="e.g., fabric, zipper, small...")

    # Apply filters
    filtered = critical_df.copy()
    if rating_filter != "All Critical":
        selected_r = int(rating_filter.split(" ")[0])
        filtered = filtered[filtered["Rating"] == selected_r]

    if dept_filter != "All Departments" and "Department Name" in filtered.columns:
        filtered = filtered[filtered["Department Name"].astype(str) == dept_filter]

    if search_query.strip():
        term = search_query.strip().lower()
        mask = filtered["clean_text"].str.contains(term, na=False)
        if "Title" in filtered.columns:
            mask = mask | filtered["Title"].astype(str).str.lower().str.contains(term, na=False)
        filtered = filtered[mask]

    st.caption(f"Showing **{len(filtered):,}** matching critical reviews")

    # Build display table with only available columns
    display_cols = ["Rating", "Review Text"]
    for opt_col in ["Title", "Department Name", "Class Name", "Clothing ID"]:
        if opt_col in filtered.columns:
            display_cols.append(opt_col)

    st.dataframe(filtered[display_cols], use_container_width=True, height=400)


# =============================================================================
# PAGE 4: TOP 3 SEVERE COMPLAINTS
# =============================================================================
elif page == "⭐ Top 3 Severe Complaints":
    st.title("⭐ Top 3 Most Critical Reviews")
    min_rating = int(df_cleaned["Rating"].min()) if len(df_cleaned) else 1
    st.markdown(f"""
    **Selection Rule:**
    > Filter for minimum rating (`Rating == {min_rating}`), then rank descending by original `Review Text` character length.

    *Why this rule?* Lowest-rated reviews represent peak customer dissatisfaction. Sorting by character length prioritizes detailed, articulate complaints providing the specific context required for empathetic apology emails.
    """)

    try:
        top_3_response = select_top_critical_reviews(df_cleaned, n=3)
    except ValueError as exc:
        st.error(f"Could not select top 3 reviews: {exc}")
        st.stop()

    st.caption(f"*Selection rule used: {top_3_response.selection_rule}*")

    for i, review in enumerate(top_3_response.reviews):
        with st.container():
            st.markdown(f"### Rank {i+1} — {review.title if review.title else '(No Title)'}")

            meta_cols = st.columns(4)
            meta_cols[0].markdown(f"**Rating:** {'★' * review.rating}{'☆' * max(0, 5 - review.rating)} ({review.rating}/5)")
            meta_cols[1].markdown(f"**Length:** {review.review_length} characters")
            meta_cols[2].markdown(f"**Department:** {review.department_name or 'N/A'}")
            meta_cols[3].markdown(f"**Product ID:** #{review.clothing_id or 'N/A'}")

            st.markdown(f"""
            <div class="review-quote">
                "{review.review_text}"
            </div>
            """, unsafe_allow_html=True)

            if review.detected_complaints:
                badges = " ".join([f"`{c}`" for c in review.detected_complaints])
                st.markdown(f"**Detected Complaint Categories:** {badges}")

            if st.button(f"Draft AI Response for Case {i+1}", key=f"btn_case_{i+1}"):
                st.session_state["selected_review_text"] = review.review_text
                st.session_state["selected_rating"] = review.rating
                st.session_state["selected_title"] = review.title
                st.session_state["selected_dept"] = review.department_name
                st.info("Loaded into AI Response Generator! Navigate to '🤖 AI Response Generator' in the sidebar.")
            st.markdown("---")


# =============================================================================
# PAGE 5: AI RESPONSE GENERATOR
# =============================================================================
elif page == "🤖 AI Response Generator":
    st.title("🤖 AI Apology Response Assistant")
    st.markdown("""
    Draft personalized, empathetic customer apology emails powered by Google Gemini.
    
    ⚠️ **Human-in-the-Loop Protocol:** *AI drafts must be reviewed and approved by human support staff before transmission.*
    """)

    try:
        top_3_response = select_top_critical_reviews(df_cleaned, n=3)
        default_text = st.session_state.get("selected_review_text", top_3_response.reviews[0].review_text)
        default_rating = st.session_state.get("selected_rating", top_3_response.reviews[0].rating)
        default_title = st.session_state.get("selected_title", top_3_response.reviews[0].title)
        has_presets = True
    except ValueError:
        default_text = ""
        default_rating = 1
        default_title = ""
        has_presets = False

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("#### Customer Review Context")

        if has_presets:
            preset_choice = st.selectbox(
                "Select Review Preset",
                ["Preset: Top Review #1", "Preset: Top Review #2", "Preset: Top Review #3", "Custom Review Input"]
            )
            if preset_choice == "Preset: Top Review #1":
                cur_r = top_3_response.reviews[0]
            elif preset_choice == "Preset: Top Review #2":
                cur_r = top_3_response.reviews[1]
            elif preset_choice == "Preset: Top Review #3":
                cur_r = top_3_response.reviews[2]
            else:
                cur_r = None

            if cur_r:
                inp_text = cur_r.review_text
                inp_rating = cur_r.rating
                inp_title = cur_r.title or ""
                inp_dept = cur_r.department_name
                st.markdown(f"**Title:** {inp_title if inp_title else '(No Title)'}")
                st.markdown(f"**Rating:** {'★' * inp_rating} ({inp_rating}/5)")
                st.markdown(f"""<div class="review-quote">"{inp_text}"</div>""", unsafe_allow_html=True)
            else:
                inp_text = st.text_area("Customer Review Text", value=default_text, height=180)
                inp_rating = st.slider("Customer Rating", 1, 5, value=default_rating)
                inp_title = st.text_input("Review Title", value=default_title)
                inp_dept = None
        else:
            st.info("No presets available — enter a custom review below.")
            inp_text = st.text_area("Customer Review Text", height=180)
            inp_rating = st.slider("Customer Rating", 1, 5, value=1)
            inp_title = st.text_input("Review Title", value="")
            inp_dept = None

        detected = detect_complaint_indicators(inp_text)
        if detected:
            st.markdown(f"**Detected Themes:** {' '.join([f'`{d}`' for d in detected])}")

        generate_clicked = st.button("🚀 Generate AI Response Draft", type="primary")

    with col_out:
        st.markdown("#### Generated Apology Draft")

        if generate_clicked:
            with st.spinner("Connecting to Google Gemini API..."):
                draft = generate_apology_email(
                    review_text=inp_text,
                    rating=inp_rating,
                    title=inp_title,
                    department=inp_dept,
                )
                st.session_state["current_draft"] = draft

        draft = st.session_state.get("current_draft", None)

        if draft:
            if draft.is_fallback:
                st.warning(f"⚠️ {draft.error_message}")

            st.text_input("Email Subject Line", value=draft.subject, key="draft_subject")
            st.text_area("Email Body", value=draft.email_body, height=220, key="draft_body")

            word_count = draft.validation.word_count
            st.markdown(f"**Word Count:** `{word_count} / 130 words` | **Model:** `{draft.model_used}`")

            st.markdown("##### Quality & Safety Validation:")
            val = draft.validation
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.markdown(f"{'✅' if val.word_limit else '❌'} Under 130 words: `{val.word_count} words`")
                st.markdown(f"{'✅' if val.signoff else '❌'} Required 'Customer Care Team' sign-off")
                st.markdown(f"{'✅' if val.next_step else '❌'} Concrete next step (refund/return/exchange)")
            with v_col2:
                st.markdown(f"{'✅' if val.no_fabricated_action else '❌'} No fabricated actions or order numbers")
                st.markdown(f"{'✅' if val.specificity else '❌'} Context-specific complaint references")
                st.markdown(f"{'✅' if val.not_empty else '❌'} Non-empty response")

            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                st.button("📋 Copy to Clipboard (Ready for Human Review)", help="Simulates sending to customer support ticketing system")
            with b2:
                if st.button("🔄 Clear / Reset"):
                    st.session_state.pop("current_draft", None)
                    st.rerun()
        else:
            st.info("Select a customer review on the left and click **'Generate AI Response Draft'** to view the personalized email.")


# =============================================================================
# PAGE 6: SAFETY & VALIDATION
# =============================================================================
elif page == "✅ Safety & Validation":
    st.title("✅ AI Safety & Guardrails Engine")
    st.markdown("""
    In enterprise customer service applications, unconstrained Generative AI can hallucinate, make unauthorized commitments, or use inappropriate tone.
    Our solution incorporates a multi-layer verification framework.
    """)

    st.markdown("#### Anti-Fabrication Rule Set")
    st.markdown("""
    The system strictly prevents the AI from falsely claiming that an action has already occurred:
    - ❌ **Forbidden:** *"I have initiated your refund."* (Falsely claims transaction occurred)
    - ❌ **Forbidden:** *"I created your prepaid shipping label."* (Invented label generation)
    - ❌ **Forbidden:** *"Your order #9821 has been cancelled."* (Fabricated order identifier)
    - ✅ **Permitted:** *"We would be happy to help arrange a return or refund for you."* (Realistic offer)
    - ✅ **Permitted:** *"Please reply so our Customer Care Team can assist you with next steps."* (Empathetic assistance)
    """)

    st.markdown("#### Interactive Guardrail Validator")
    st.caption("Test the automated validation engine with any custom email draft.")

    test_input = st.text_area(
        "Enter Draft Response to Validate",
        value=(
            "Dear Valued Customer,\n\n"
            "Thank you for sharing your feedback. I am so sorry that the fabric and fit of your dress "
            "did not meet your expectations. We would be happy to help arrange a return or refund for you. "
            "Please reply to this message and our team will guide you through the process.\n\n"
            "Warm regards,\nCustomer Care Team"
        ),
        height=160
    )

    if st.button("Run Safety Validation Audit"):
        res = validate_apology_response(test_input, "dress was poorly made and fabric see through", 1)
        st.markdown(f"### Overall Audit Status: {'✅ PASSED ALL CHECKS' if res.passed_all else '❌ FAILED CHECKS'}")

        for rule, msg in res.details.items():
            status = getattr(res, rule, False)
            if status:
                st.markdown(f"- ✅ **{rule.replace('_', ' ').title()}:** {msg}")
            else:
                st.markdown(f"- ❌ **{rule.replace('_', ' ').title()}:** {msg}")


# =============================================================================
# PAGE 7: ABOUT & METHODOLOGY
# =============================================================================
elif page == "ℹ️ About / Methodology":
    st.title("ℹ️ Methodology & Architecture")
    st.markdown("""
    ### Imarticus Data Science Internship Assessment
    **Customer Feedback Intelligence & AI Response System**
    
    #### 1. Assessment Workflow
    This project demonstrates the transition from exploratory data science to a production-grade enterprise decision-support tool.
    
    ```
    Customer Reviews CSV  (official OR compatible dataset)
            ↓
    Smart Schema Detection  (50+ column aliases, auto-mapping)
            ↓
    Deterministic Pandas Cleaning
            ↓
    Rule-Based Critical Filter (Rating ≤ threshold)
            ↓
    Keyword & Bigram Counter Analysis
            ↓
    Deterministic Top-3 Severe Selection (min rating, longest text)
            ↓
    Google Gemini Apology Draft Generation
            ↓
    Automated Safety & Quality Validation
            ↓
    Human Support Staff Review & Dispatch
    ```

    #### 2. Smart Dataset Support
    The application accepts **any customer-review CSV** with at least a review text and rating column, using any of 50+ recognised column name aliases.  The official Imarticus Women's Clothing dataset is auto-detected by its schema fingerprint.

    #### 3. Why Rule-Based Filtering instead of ML?
    1. **100% Deterministic & Auditable:** Customer support teams require clear, unambiguous criteria.
    2. **Zero False Positives from Model Drift:** Ratings of 1 and 2 stars are explicitly defined negative experiences.
    3. **No Training/Annotation Overhead:** Enables instant cold-start operation on any e-commerce dataset.

    #### 4. Technology Stack
    - **Language:** Python 3.11
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly, Matplotlib
    - **Generative AI:** Google GenAI SDK (`google-genai`), Gemini 2.5 / 3.1 / 3.8
    - **UI Dashboard:** Streamlit
    - **Testing:** Pytest (36 passing unit tests)
    """)
