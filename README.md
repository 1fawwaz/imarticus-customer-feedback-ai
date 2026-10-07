# Customer Feedback Intelligence & AI Response System

An enterprise decision-support tool and analytical system designed for the **Imarticus Data Science Internship Assessment (Python Foundations & Gen AI)**.

Built on the Kaggle *Women's E-Commerce Clothing Reviews* dataset, this project demonstrates end-to-end data wrangling, deterministic rule-based critical feedback triage, complaint keyword intelligence, and Generative AI apology email drafting using Google Gemini with strict safety guardrails and a human-in-the-loop workflow.

---

## 1. Project Overview

Retail e-commerce organizations receive thousands of product reviews weekly. Manual triage of negative feedback is labor-intensive, error-prone, and slow. 

This project solves that challenge by providing:
1. **Automated Data Cleaning:** Standardizes raw feedback, coerces data types, and normalizes text.
2. **Rule-Based Critical Triage:** Deterministically isolates critical customer feedback (`Rating <= 2`) without black-box machine learning drift.
3. **Complaint Intelligence:** Discovers root causes using `collections.Counter` unigrams, bigrams, and predefined problem categories (`fit`, `fabric`, `small`, `see-through`, etc.).
4. **Deterministic Top-3 Escalation:** Identifies the 3 most detailed, severe 1-star complaints (`Rating == 1` sorted by longest text).
5. **AI Apology Drafting (Google Gemini):** Generates empathetic, personalized customer apology drafts under 130 words with concrete next steps (refund, exchange, or return).
6. **Strict Safety & Anti-Fabrication Guardrails:** Validates that the AI never falsely claims an action has already occurred (e.g., *"I have initiated your refund"* is strictly forbidden; *"We would be happy to help arrange a return or refund"* is required).
7. **Interactive Streamlit Application:** A production dashboard providing KPI metric cards, interactive filters, complaint visualizations, and an AI drafting workbench.

---

## 2. Business Problem & Solution Architecture

### The Business Challenge
- Customer support agents spend hours manually reading negative reviews.
- Standard boilerplate replies feel robotic and impersonal, worsening customer dissatisfaction.
- Unconstrained Generative AI poses significant liability if it hallucinates false order numbers or unauthorized refund promises.

### The Solution: Human-in-the-Loop Decision Support
The system acts as an **assistive drafting tool**, never sending emails autonomously:

```
                  Raw Customer Reviews CSV
                             │
                             ▼
               Deterministic Data Cleaning
                 (Pandas, Regex, Deduplication)
                             │
                             ▼
               Rule-Based Critical Filtering
                       (Rating <= 2)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     Complaint Intelligence          Top 3 Severe Queue
   (Keywords, Bigrams, Terms)    (Rating == 1, Longest Text)
              │                             │
              ▼                             ▼
       Executive Insights          Google Gemini Drafts
                                (google-genai, SDK)
                                            │
                                            ▼
                                Safety Guardrail Audit
                            (Word count, Sign-off, Ethics)
                                            │
                                            ▼
                                Human Support Review
                              (Approval / Manual Send)
```

---

## 3. Dataset & Cleaning Methodology

### Dataset Details
- **Source:** Kaggle *Women's E-Commerce Clothing Reviews*
- **Path:** `data/Womens Clothing E-Commerce Reviews.csv`
- **Raw Dimensions:** 23,486 rows × 11 columns
- **Expected Columns:** `Unnamed: 0`, `Clothing ID`, `Age`, `Title`, `Review Text`, `Rating`, `Recommended IND`, `Positive Feedback Count`, `Division Name`, `Department Name`, `Class Name`.

### Cleaning Rules
1. **Index Artifact Removal:** Drop `Unnamed: 0` column.
2. **Type Coercion:** Convert `Rating` to numeric, coercing non-numeric values to `NaN`.
3. **Whitespace Normalization:** Convert blank and whitespace-only reviews to `NaN`.
4. **Missing Value Filtering:** Drop rows missing `Review Text` or `Rating`.
5. **Title Imputation:** Impute missing `Title` values with empty string `""`.
6. **Range Verification:** Validate that `Rating` is strictly between `1` and `5`.
7. **Deduplication:** Remove exact duplicate records (`drop_duplicates()`).
8. **Text Normalization:** Create `clean_text` column (lowercased, stripped digits/special characters, whitespace normalized) while **strictly preserving original `Review Text`** for contextual AI drafting.

**Data Metrics (Official Assessment Dataset):**
- Raw rows: **23,486**
- Cleaned rows: **22,640**
- Rows removed: **846** (missing text/ratings and duplicates)

---

### 3.1 Smart Schema Detection & Universal CSV Support

Beyond the official dataset, the application dynamically accepts **any customer review CSV** via `src/schema_detection.py`:

- **Automatic Fingerprint Recognition:** If the CSV matches the Imarticus Women's Clothing review columns, it is tagged `🟢 Imarticus Assessment Dataset Detected`.
- **Flexible Column Mapping:** Detects 15+ aliases for review text (`review text`, `customer_review`, `comment`, `feedback`, `body`, `review_content`, etc.) and 15+ aliases for rating (`rating`, `score`, `stars`, `overall`, `star_rating`, etc.), case- and whitespace-insensitively.
- **Graceful Optional Column Handling:** Columns such as Department, Clothing ID, Title, and Age are mapped if present; missing optional columns degrade gracefully without breaking any visual or analytical component.
- **Configurable Critical Threshold:** For non-standard scales (e.g., 1–10), users can adjust the critical triage threshold dynamically in the sidebar.
- **Graceful Rejection:** Unrelated datasets (e.g. employee rosters, financial sheets) are rejected with clear, friendly diagnostic messages indicating required columns.

---

## 4. Rule-Based Critical Filtering

The assessment strictly mandates deterministic logic rather than machine learning:
$$\text{Critical Review} \iff \text{Rating} \le 2$$

- **Total Cleaned Reviews:** 22,640
- **Critical Reviews:** 2,370 (**10.47%**)
- **1-Star Reviews:** 821
- **2-Star Reviews:** 1,549

*Why rule-based?*
1. **100% Auditable:** Transparent criteria for support routing and auditing.
2. **Zero False Negatives:** Dissatisfied 1-star customers are never missed due to probabilistic thresholding.
3. **Cold-Start Ready:** Operates instantly on any product category without training data.

---

## 5. Complaint Intelligence & Bigrams

Using Python's `collections.Counter`, the system extracts:
- **Top Complaint Keywords:** `dress`, `size`, `like`, `fit`, `fabric`, `small`, `color`, `material`
- **Predefined Term Frequencies:**
  - `fit`: 590
  - `fabric`: 579
  - `small`: 413
  - `color`: 335
  - `large`: 260
  - `returned`: 168
  - `cheap`: 155
  - `see-through`: 65 *(handles both hyphenated and spaced forms)*
  - `poor`: 61
  - `itchy`: 46
  - `refund`: 5
- **Top Bigrams:** `poor quality`, `see through`, `too small`, `too large`, `looked like`, `wanted love`

---

## 6. Deterministic Selection of Top 3 Reviews

To identify candidates for personalized apology generation, the system applies:
1. Filter `Rating == 1`.
2. Rank descending by character length of the original `Review Text`.
3. Select the top 3 records.

**Selected Reviews:**
- **Case 1:** Length = 508 characters | Title: *"I don't understand this dress"* (Complaints: Matronly cut, hot pink belt)
- **Case 2:** Length = 506 characters | Title: *"Never been more disappointed..."* (Complaints: Severe sweater quality defect)
- **Case 3:** Length = 504 characters | Title: *"So small, so sad."* (Complaints: Sizing defect, button couldn't meet buttonhole)

---

## 7. Generative AI & Safety Guardrails

### Model Configuration
- **Default Assessment Model:** `gemini-2.5-flash`
- **Environment Override:** `GEMINI_MODEL=gemini-3.1-flash-lite` or `gemini-3.8-flash` via `.env` to accommodate Google API account availability.
- **SDK:** Official Google GenAI SDK (`google-genai`).

### System Prompt & Anti-Fabrication Rules
```text
You are a Customer Support Agent for a retail clothing company. Write a short (under 130 words), warm, personalized, empathetic apology email. Reference the specific complaints in the review, avoid generic wording, offer a concrete next step (refund, exchange or free return), and sign off as 'Customer Care Team'. Do not invent order numbers, customer information, completed actions, or company policies. Never claim that an action (such as issuing a refund or generating a shipping label) has already been processed or initiated; instead offer to arrange it once the customer replies.
```

### Safety Validation Engine
Every draft is audited before presentation:
| Check | Requirement | Purpose |
|---|---|---|
| **Word Limit** | $\le 130$ words | Ensures concise, customer-friendly communication |
| **Sign-Off** | `Customer Care Team` | Standardizes brand identity |
| **Specificity** | Contains review keywords | Prevents generic, dismissive responses |
| **Actionable Next Step** | Offers refund/return/exchange | Gives clear resolution pathway |
| **No Fabricated Action** | Disallows *"I have initiated..."* | Prevents unauthorized commitments |
| **Non-Empty** | Real text generated | Verifies API success |

---

## 8. Project Structure

```
imarticus_customer_feedback_project/
│
├── app.py                             # Interactive Streamlit application
├── customer_feedback_analysis.ipynb   # Official assessment Jupyter notebook
│
├── src/                               # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                      # Paths, models, stopwords, system prompt
│   ├── schemas.py                     # Pydantic data models & transfer schemas
│   ├── schema_detection.py            # Smart CSV schema detection & alias mapping
│   ├── cleaning.py                    # Deterministic data cleaning pipeline
│   ├── analysis.py                    # Dataset overview & query filtering
│   ├── complaint_analysis.py          # Counter keywords, bigrams & terms
│   ├── review_selection.py            # Deterministic top-3 review selector
│   ├── gemini_service.py              # Google GenAI SDK integration & retry
│   ├── validators.py                  # Guardrails & safety validator
│   └── automation.py                  # Batch pipeline generating outputs
│
├── tests/                             # Comprehensive test suite (36 passing tests)
│   ├── __init__.py
│   ├── test_schema_detection.py       # 22 schema detection & multi-dataset tests
│   ├── test_cleaning.py               # Missing data, deduplication, clean_text
│   ├── test_analysis.py               # Rule-based filter & indicator detection
│   ├── test_complaint_analysis.py     # Keywords, bigrams & see-through tests
│   ├── test_review_selection.py       # Top-3 ranking & tie-break logic
│   ├── test_validators.py             # Word limit, signoff & anti-fabrication
│   └── test_gemini_service.py         # Fallback & error handling tests
│
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv  # 8.48 MB Kaggle dataset
│
├── outputs/                           # Batch pipeline artifacts
│   ├── critical_reviews.csv
│   ├── complaint_summary.csv
│   └── ai_response_drafts.csv
│
├── requirements.txt                   # Locked project dependencies
├── .env.example                       # Environment template
├── .env                               # Local secrets (gitignored)
└── .gitignore                         # Excludes secrets, venv, and cache
```

---

## 9. Quickstart & Installation

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Setup Virtual Environment
```powershell
# Clone or navigate to the project root
git clone https://github.com/1fawwaz/imarticus-customer-feedback-ai.git
cd imarticus-customer-feedback-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
```

---

## 10. Running the Project

### Option A: Interactive Streamlit Application
Launch the customer feedback dashboard:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Streamlit Pages:**
1. 🏠 **Overview Dashboard:** High-level metrics, rating distribution, department share.
2. 📊 **Complaint Intelligence:** Top keywords, bigrams, complaint term frequency chart, and business insights.
3. 🚨 **Critical Reviews Queue:** Filterable review table with text search and department filters.
4. ⭐ **Top 3 Severe Complaints:** Articulated complaints ranked by character length.
5. 🤖 **AI Response Generator:** Generate personalized drafts, view word count, and check validation.
6. ✅ **Safety & Validation:** Interactive safety guardrail auditor.
7. ℹ️ **About / Methodology:** Architecture and educational notes.

### Option B: Official Jupyter Notebook
Run the complete assessment top to bottom:
```powershell
jupyter notebook customer_feedback_analysis.ipynb
```
Select the `.venv` kernel and choose **Kernel -> Restart & Run All**.

### Option C: Automated Batch Pipeline
Execute the headless batch processing workflow:
```powershell
python -m src.automation
```
Outputs will be written to `outputs/`.

---

## 11. Running the Test Suite

Run the full automated test suite:
```powershell
pytest -v
```
**Expected Output:**
```text
tests/test_analysis.py::test_filter_critical_reviews_rule_based PASSED
tests/test_analysis.py::test_detect_complaint_indicators PASSED
tests/test_cleaning.py::test_clean_review_text PASSED
tests/test_cleaning.py::test_clean_dataset_handles_missing_and_duplicates PASSED
tests/test_complaint_analysis.py::test_complaint_keywords_and_bigrams PASSED
tests/test_complaint_analysis.py::test_predefined_terms_handles_see_through PASSED
tests/test_gemini_service.py::test_extract_subject_and_body PASSED
tests/test_gemini_service.py::test_gemini_service_handles_missing_api_key PASSED
tests/test_review_selection.py::test_select_top_critical_reviews_ranking PASSED
tests/test_review_selection.py::test_select_top_critical_reviews_insufficient_records PASSED
tests/test_validators.py::test_validator_passes_compliant_response PASSED
tests/test_validators.py::test_validator_fails_fabricated_action PASSED
tests/test_validators.py::test_validator_fails_missing_signoff PASSED
tests/test_validators.py::test_validator_fails_word_count_exceeded PASSED

============================= 14 passed in 0.55s =============================
```

---

## 12. Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub (ensure `.env` is **NOT** committed; it is protected by `.gitignore`).
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository and set the main file path to `app.py`.
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
5. Deploy.

---

## 13. Limitations & Future Improvements

### Current Limitations
- **Binary Filtering:** Dissatisfied customers rating 3 stars with negative text are excluded by the strict `Rating <= 2` assessment rule.
- **Syntactic Keyword Matching:** `Counter` relies on exact token matching and does not capture latent semantic synonyms.
- **Quota Constraints:** Free-tier Gemini endpoints have RPM/TPM limits.

### Realistic Future Improvements
- **Semantic Aspect Clustering:** Group complaints using topic modeling (e.g., LDA or BERTopic).
- **Automated CRM/Zendesk Integration:** Push approved drafts directly into customer support ticketing queues.
- **Multilingual Support:** Auto-detect review language and generate localized apologies.
- **Sentiment Shift Tracking:** Monitor rolling 30-day complaint frequencies to alert QA teams of product batch defects.

---

## 14. Author & Certification

Built for the **Imarticus Data Science Internship Assessment — Python Foundations & Gen AI**.
Demonstrating clean Python architecture, reproducible data science, Generative AI prompt engineering, and human-in-the-loop system design.
