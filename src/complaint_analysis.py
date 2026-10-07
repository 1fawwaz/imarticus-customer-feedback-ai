"""Complaint keyword analysis — find the most common complaint words and phrases."""

import re
from collections import Counter
from typing import List, Optional
import pandas as pd

from src.config import MIN_WORD_LEN, CUSTOM_STOPWORDS, COMPLAINT_TERMS


def get_top_keywords(df: pd.DataFrame, n: int = 15) -> List[dict]:
    """Find the most frequent meaningful words in critical review text.
    
    Skips short words and common stopwords so the results are actually useful.
    Returns a list of dicts: [{"term": word, "count": frequency}, ...]
    """
    counter = Counter()
    for text in df["clean_text"].dropna():
        tokens = text.split()
        valid = [t for t in tokens if len(t) >= MIN_WORD_LEN and t not in CUSTOM_STOPWORDS]
        counter.update(valid)
    return [{"term": word, "count": int(count)} for word, count in counter.most_common(n)]


def get_top_bigrams(df: pd.DataFrame, n: int = 10) -> List[dict]:
    """Find the most common 2-word phrases in critical reviews.
    
    Bigrams help identify complaint patterns like 'wrong size' or 'bad quality'.
    Returns a list of dicts: [{"phrase": bigram, "count": frequency}, ...]
    """
    counter = Counter()
    for text in df["clean_text"].dropna():
        tokens = text.split()
        if len(tokens) >= 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            counter.update(bigrams)
    return [{"phrase": phrase, "count": int(count)} for phrase, count in counter.most_common(n)]


def get_predefined_term_counts(df: pd.DataFrame) -> List[dict]:
    """Count how many critical reviews mention each of the predefined complaint words.
    
    These are specific terms we're tracking: fit, fabric, color, size, etc.
    Returns a sorted list (highest count first): [{"term": ..., "count": ...}, ...]
    """
    results = []
    for term in COMPLAINT_TERMS:
        if term == "see-through":
            pattern = r"\bsee[\s\-]through\b"
            count = int(df["reviewText"].astype(str).str.contains(pattern, case=False, regex=True).sum())
        else:
            pattern = rf"\b{re.escape(term)}\b"
            count = int(df["clean_text"].astype(str).str.contains(pattern, case=False, regex=True).sum())
        results.append({"term": term, "count": count})

    results.sort(key=lambda x: x["count"], reverse=True)
    return results


def get_keyword_insights(df: pd.DataFrame) -> dict:
    """Run all keyword analyses and return the combined results.
    
    Returns a dict with:
    - top_keywords: most frequent words in critical reviews
    - predefined_terms: counts for our tracked complaint words
    - top_bigrams: most common 2-word phrases
    """
    return {
        "top_keywords": get_top_keywords(df, n=15),
        "predefined_terms": get_predefined_term_counts(df),
        "top_bigrams": get_top_bigrams(df, n=10),
    }
