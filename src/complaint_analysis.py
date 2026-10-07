"""Complaint keyword, bigram, and specific term frequency analysis module."""

import re
from collections import Counter
from typing import List, Optional, Set
import pandas as pd

from src.config import MIN_WORD_LEN, CUSTOM_STOPWORDS, PREDEFINED_COMPLAINT_TERMS
from src.schemas import ComplaintTermFrequency, BigramFrequency, KeywordInsights


def get_top_complaint_keywords(
    df: pd.DataFrame,
    n: int = 15,
    min_len: int = MIN_WORD_LEN,
    stopwords: Optional[Set[str]] = None,
) -> List[ComplaintTermFrequency]:
    """Calculate the most frequent complaint words using collections.Counter.

    Filters tokens by minimum length and stopwords.
    """
    if stopwords is None:
        stopwords = CUSTOM_STOPWORDS

    counter: Counter = Counter()

    for text in df["clean_text"].dropna():
        tokens = text.split()
        valid_tokens = [
            t for t in tokens
            if len(t) >= min_len and t not in stopwords
        ]
        counter.update(valid_tokens)

    return [
        ComplaintTermFrequency(term=word, count=int(count))
        for word, count in counter.most_common(n)
    ]


def get_top_bigrams(df: pd.DataFrame, n: int = 10) -> List[BigramFrequency]:
    """Calculate the top adjacent word pairs in the cleaned reviews."""
    bigram_counter: Counter = Counter()

    for text in df["clean_text"].dropna():
        tokens = text.split()
        if len(tokens) >= 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            bigram_counter.update(bigrams)

    return [
        BigramFrequency(phrase=phrase, count=int(count))
        for phrase, count in bigram_counter.most_common(n)
    ]


def get_predefined_term_counts(
    df: pd.DataFrame,
    terms: Optional[List[str]] = None,
) -> List[ComplaintTermFrequency]:
    """Count occurrences of specific predefined complaint keywords in review text.

    Supports both 'see-through' and 'see through' for robust detection.
    """
    if terms is None:
        terms = PREDEFINED_COMPLAINT_TERMS

    counts: List[ComplaintTermFrequency] = []

    for term in terms:
        if term == "see-through":
            # Check for hyphenated or space-separated variations
            pattern = r"\bsee[\s\-]through\b"
            matches = df["Review Text"].astype(str).str.contains(pattern, case=False, regex=True).sum()
        else:
            pattern = rf"\b{re.escape(term)}\b"
            matches = df["clean_text"].astype(str).str.contains(pattern, case=False, regex=True).sum()

        counts.append(ComplaintTermFrequency(term=term, count=int(matches)))

    # Sort descending by frequency
    counts.sort(key=lambda x: x.count, reverse=True)
    return counts


def get_keyword_insights(df: pd.DataFrame) -> KeywordInsights:
    """Generate comprehensive complaint keyword, bigram, and predefined term insights."""
    top_kw = get_top_complaint_keywords(df, n=15)
    predefined = get_predefined_term_counts(df)
    top_bg = get_top_bigrams(df, n=10)

    return KeywordInsights(
        top_keywords=top_kw,
        predefined_terms=predefined,
        top_bigrams=top_bg,
    )
