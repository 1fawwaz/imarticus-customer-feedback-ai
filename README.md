# Customer Feedback Analysis and AI Response

An end-to-end data science project that analyzes women's clothing e-commerce reviews to identify critical customer complaints and generate AI-powered apology email drafts using Google Gemini.

Built as part of the **Imarticus Data Science Internship Assessment**.

---

## Objective

The goal of this project is to:
1. Load and clean a real customer review dataset using Pandas
2. Identify "critical" reviews (rating ≤ 2) using a simple rule-based filter
3. Analyze what customers are complaining about using keyword frequency analysis
4. Select the top 3 most detailed critical reviews
5. Use Google Gemini to generate personalized apology email drafts
6. Validate each draft against a set of quality rules

---

## Dataset

**Women's Clothing E-Commerce Reviews** (sourced from Kaggle)

The dataset contains 23,486 reviews. After cleaning, 22,640 reviews remain.

Columns used in this project:

| Column | Description |
|--------|-------------|
| `reviewText` | The full customer review |
| `overall` | Star rating (1–5) |
| `summary` | Short review title |
| `asin` | Product ID |
| `helpful` | Number of helpful votes |

**Key result:** 2,370 critical reviews (10.47% of cleaned data).

---

## Tools Used

- **Python** — core language
- **Pandas** — data loading and cleaning
- **collections.Counter** — keyword frequency analysis
- **Streamlit** — interactive web app
- **Plotly** — charts and visualizations
- **Google Gemini API** — AI email generation
- **python-dotenv** — API key management
- **pytest** — unit tests

---

## Steps

### 1. Data Cleaning
- Load the CSV and rename columns to standard names
- Drop rows with missing review text or rating
- Remove duplicates
- Keep only ratings between 1 and 5
- Create a `clean_text` column (lowercase, letters only) for keyword analysis

### 2. Critical Review Filtering
- A review is "critical" if `overall <= 2`
- This is a simple, auditable rule — no machine learning involved

### 3. Complaint Keyword Analysis
- Count predefined complaint terms: *fit, fabric, color, size, see-through, poor, cheap, itchy, refund, returned, large, small*
- Find the top 15 most frequent words (stopwords removed)
- Extract common 2-word phrases (bigrams) using a sliding window

### 4. Top 3 Review Selection
- Filter for the lowest-rated reviews (usually 1-star)
- Sort by review text length (longest first)
- Pick the top 3 — longer reviews have more specific complaints, leading to better AI responses

### 5. AI Response Generation (Google Gemini)
- Send each review to the Gemini API with a structured prompt
- The AI is instructed to write under 130 words, offer a next step, and sign off as 'Customer Care Team'
- Automatic retry logic handles API timeouts

### 6. Response Validation
Each generated email is automatically checked for:
- Word count ≤ 130
- Contains 'Customer Care Team' sign-off
- Offers a refund, return, or exchange
- Does NOT falsely claim an action was already completed
- References specific words from the original review

---

## Results

| Metric | Value |
|--------|-------|
| Raw rows | 23,486 |
| Cleaned rows | 22,640 |
| Critical reviews (≤ 2★) | 2,370 |
| Critical review rate | 10.47% |
| Top complaint term | fit |

---

## Streamlit App

The app has 7 sections:

1. **Overview** — key metrics and rating distribution chart
2. **Complaint Analysis** — keyword frequency, bigrams, tracked terms
3. **Critical Reviews** — searchable/filterable table of all critical reviews
4. **Top 3 Reviews** — the 3 most detailed 1-star reviews
5. **AI Response Generator** — generate and edit Gemini apology drafts
6. **Validation** — check any email draft against quality rules
7. **Methodology** — how the project works

🌐 **Live App:** https://1fawwaz-imarticus-customer-feedback-ai-app-bovlj8.streamlit.app/

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/1fawwaz/imarticus-customer-feedback-ai.git
cd imarticus-customer-feedback-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash
```

> If no API key is set, the app runs in fallback mode with a template response.

### 4. Add the dataset

Place the CSV in the `data/` folder:

```
data/Womens Clothing E-Commerce Reviews.csv
```

### 5. Run the app

```bash
streamlit run app.py
```

### 6. (Optional) Run the batch pipeline

This processes the full dataset and saves results to `outputs/`:

```bash
python -m src.automation
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
imarticus_customer_feedback_project/
│
├── customer_feedback_analysis.ipynb   # Jupyter notebook (assessment)
├── app.py                             # Streamlit web app
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv
│
├── src/
│   ├── config.py               # Settings and constants
│   ├── cleaning.py             # Data loading and cleaning
│   ├── analysis.py             # Filtering and overview stats
│   ├── complaint_analysis.py   # Keyword and bigram analysis
│   ├── review_selection.py     # Top 3 review selection
│   ├── gemini_service.py       # Gemini API integration
│   ├── validators.py           # Response quality checks
│   └── automation.py           # Batch pipeline script
│
└── tests/
    ├── test_cleaning.py
    ├── test_analysis.py
    ├── test_complaint_analysis.py
    ├── test_review_selection.py
    ├── test_validators.py
    └── test_pipeline.py
```

---

## Future Improvements

- Add sentiment analysis on top of the rule-based filter
- Let users export AI drafts directly as a CSV or PDF
- Add a confidence score to the keyword analysis
- Support batch processing of all critical reviews through the UI

---

## Links

- 🔗 GitHub: https://github.com/1fawwaz/imarticus-customer-feedback-ai
- 🌐 Live App: https://1fawwaz-imarticus-customer-feedback-ai-app-bovlj8.streamlit.app/
