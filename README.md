# Customer Feedback Intelligence & AI Response System

This project analyzes customer reviews using Python and Pandas, identifies critical negative reviews using rating-based rules, analyzes common complaint keywords, and uses Google Gemini to generate personalized apology-response drafts.

---

## Project Objective

The goal of this project is to build an automated, transparent pipeline that helps support teams handle negative customer feedback effectively. 

Specifically, the system:
1. Loads and cleans customer reviews using Pandas.
2. Identifies critical negative reviews using a clear rating-based rule (`overall <= 2`).
3. Discovers common complaint themes and keywords using frequency analysis.
4. Identifies the most detailed 1-star reviews for priority handling.
5. Generates personalized apology drafts with Google Gemini.
6. Validates generated responses against defined customer service guidelines before human review.

---

## Dataset

The project uses customer reviews with the following 5 columns:

| Column | Meaning | Description |
|--------|---------|-------------|
| `reviewText` | Customer review | The full written customer review text |
| `overall` | Rating | Customer rating on a 1 to 5 scale |
| `summary` | Review title | Short headline or title of the review |
| `asin` | Product ID | Unique product identifier |
| `helpful` | Helpful vote/count | Number of positive feedback votes received |

---

## Tools Used

- **Python** (version 3.10+) — Core programming language
- **Pandas** — Data loading, cleaning, filtering, and summary statistics
- **Streamlit** — Interactive multi-page web application
- **Plotly** — Interactive rating and complaint distribution charts
- **Google Gemini** (`google-genai`) — AI apology email draft generation
- **Pytest** — Automated test suite for cleaning, analysis, and validation logic

---

## Project Workflow

```
CSV
 ↓
Pandas Cleaning
 ↓
Critical Reviews
 ↓
Complaint Keyword Analysis
 ↓
Top Critical Reviews
 ↓
Gemini AI Response
 ↓
Validation
 ↓
Human Review
```

1. **CSV:** Read dataset from disk or file upload.
2. **Pandas Cleaning:** Remove missing values, handle duplicates, normalize text.
3. **Critical Reviews:** Filter reviews where rating is 1 or 2 stars (`overall <= 2`).
4. **Complaint Keyword Analysis:** Extract common complaint words, bigrams, and tracked terms using `collections.Counter`.
5. **Top Critical Reviews:** Select the longest 1-star reviews (`overall == 1`).
6. **Gemini AI Response:** Generate an apology draft referencing the customer's specific complaints.
7. **Validation:** Automatically evaluate draft length, sign-off, next step, and tone.
8. **Human Review:** Support representative reviews and approves the draft before sending.

---

## Data Cleaning

Data cleaning is handled in `src/cleaning.py` through `clean_dataset()`:

- **Missing `reviewText`**: Rows with missing or blank review text are removed (845 rows).
- **Missing `overall`**: Rows with missing rating values are removed (0 rows).
- **Missing `summary`**: Missing summary titles are filled with an empty string `""`.
- **Valid Ratings**: Ratings are converted to numeric format, and only valid 1–5 ratings are kept.
- **Duplicate Removal**: Exact duplicate rows are identified and removed (1 duplicate row).
- **Text Normalization (`clean_text`)**: Creates a lowercase, letters-only version of the text for keyword and bigram frequency analysis.
- **Original Text Preservation**: The original `reviewText` is strictly preserved for customer context and AI prompting.

**Cleaning Results on Current Dataset:**
- Raw rows: **23,486**
- Cleaned rows: **22,640**
- Rows removed: **846** (3.60%)

---

## Critical Review Analysis

Critical reviews are identified using a rule-based threshold:

$$\text{Critical Review} \iff \text{overall} \le 2$$

- No arbitrary machine learning model or black-box threshold is required.
- **Critical review count:** **2,370** reviews (**10.47%** of cleaned data).
- **1-Star reviews:** 821
- **2-Star reviews:** 1,549
- **3-Star reviews:** 2,823
- **4-Star reviews:** 4,908
- **5-Star reviews:** 12,539

---

## Complaint Analysis

Complaint analysis in `src/complaint_analysis.py` uses `collections.Counter` on the cleaned text:

1. **Predefined Complaint Terms**: Counts occurrences of common retail issue keywords:
   - `fit`, `size`, `fabric`, `color`, `small`, `large`, `cheap`, `poor`, `returned`, `refund`, `itchy`, `see-through`
2. **Top Keywords**: Identifies the most frequent words in critical reviews after removing common stopwords (minimum word length: 3 characters).
3. **Bigrams**: Extracts consecutive two-word phrases to highlight contextual patterns (e.g., *"poor quality"*, *"too small"*, *"see through"*).

---

## Top Critical Reviews

To select the most valuable reviews for drafting apology responses, `select_top_3_reviews()` in `src/review_selection.py` uses:

$$\text{overall} == 1 \quad \text{sorted by } \operatorname{len}(\text{reviewText}) \text{ descending}$$

Longer reviews contain specific, detailed customer feedback, providing Gemini with rich context for relevant, empathetic drafts.

The top reviews display:
- `summary`
- `overall`
- `reviewText`
- `asin`
- `helpful`

---

## Gemini AI Response Generation

In `src/gemini_service.py`, Google Gemini receives only actual dataset fields:
- `reviewText`
- `overall`
- `summary`
- `asin`

The prompt enforces strict quality and safety guidelines:
- **Length limit:** Must be under 130 words.
- **Empathetic & Personalized:** Directly addresses the customer's stated dissatisfaction.
- **Specific Next Step:** Offers a refund, return, or exchange.
- **Official Sign-off:** Concludes with `"Customer Care Team"`.
- **No False Claims:** Never claims an action has already been processed or completed (e.g., *"I have processed your refund"*).
- **No Hallucinations:** Does not invent order numbers, refund amounts, or store policies.
- **Fallback Support:** If no API key is provided or the API is unavailable, a safe template draft is generated.

---

## AI Response Validation

In `src/validators.py`, every generated email draft is validated against 6 automated criteria before human review:

1. **Not Empty**: Response contains meaningful text.
2. **Word Count**: Total words $\le 130$.
3. **Sign-off**: Contains the `"Customer Care Team"` signature.
4. **Actionable Next Step**: Contains clear resolution terms (`refund`, `return`, `exchange`, etc.).
5. **No Fabricated Actions**: Flags forbidden completed-action claims (e.g., *"we have initiated"*, *"refund has been issued"*).
6. **Specificity**: Verifies vocabulary overlap between the review and response to ensure personalization.

---

## Streamlit Application

The interactive web application provides a clean 6-page interface:

1. **📊 Overview**: Summary metrics (raw rows, clean rows, critical reviews, critical rate), rating distribution bar chart, top complaint terms chart, and cleaning breakdown.
2. **🔍 Complaint Analysis**: Interactive tabs for predefined complaint terms, top frequent keywords, and common bigrams.
3. **🚨 Critical Reviews**:
   - **⭐ Top Critical Reviews**: Displays the longest 1-star reviews with product ID, rating, full text, and a one-click button to load into the AI generator.
   - **📋 All Critical Reviews Queue**: Filterable table by rating (1 or 2 stars), product ID (`asin`), and keyword search.
4. **🤖 AI Response Generator**: Load top reviews or input custom review details, generate apology drafts with Gemini, edit the text, and view real-time validation badges.
5. **✅ Validation**: An interactive test bench to validate any draft response against the 6 quality checks.
6. **ℹ️ Methodology**: Documentation detailing the dataset schema, workflow, rules, and guidelines.

---

## How to Run

### 1. Prerequisites

Python 3.10 or higher installed on your machine.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Add your Gemini API key inside `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

*(If you do not set an API key, the application runs smoothly using safe fallback response templates.)*

### 4. Ensure Dataset Is in Place

Ensure the CSV dataset is placed in the `data/` directory:

```
data/Womens Clothing E-Commerce Reviews.csv
```

### 5. Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Project Structure

```
imarticus_customer_feedback_project/
│
├── customer_feedback_analysis.ipynb   # Jupyter analysis notebook
├── app.py                             # Streamlit 6-page web application
├── README.md                          # Project documentation
├── requirements.txt                   # Project dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv   # Dataset
│
├── src/
│   ├── config.py                      # Constants, file paths, complaint terms
│   ├── cleaning.py                    # Pandas cleaning and normalization
│   ├── analysis.py                    # Critical review filtering and overview stats
│   ├── complaint_analysis.py          # Keyword, bigram, and term counting
│   ├── review_selection.py            # Top 1-star critical review selection
│   ├── gemini_service.py              # Google Gemini API integration
│   └── validators.py                  # Automated response quality checks
│
└── tests/
    ├── test_cleaning.py               # Cleaning and normalization tests
    ├── test_analysis.py               # Filtering and query tests
    ├── test_complaint_analysis.py     # Keyword and bigram tests
    ├── test_review_selection.py       # Top 3 review selection tests
    ├── test_validators.py             # Response validation rule tests
    └── test_pipeline.py               # End-to-end integration tests
```

---

## Testing

The test suite contains **55 automated unit and integration tests** covering all data processing, filtering, selection, and validation logic.

To run the full test suite:

```bash
pytest tests/ -v
```

Expected output:
```
============================== 55 passed in 0.77s ==============================
```

---

## Future Improvements

- Add customer sentiment progression tracking over time.
- Implement one-click export for approved apology drafts to CSV or customer support tickets.
- Add multi-language translation support for international customer reviews.
- Introduce customizable apology tone presets (formal, conversational, expedited).
