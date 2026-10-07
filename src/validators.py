"""Validate AI-generated apology emails against basic quality rules."""

import re
from typing import Dict

# Phrases that falsely claim an action was already completed
FABRICATED_ACTION_PATTERNS = [
    r"\bi have initiated\b",
    r"\bi have processed\b",
    r"\bi have issued\b",
    r"\bi have refunded\b",
    r"\byour refund has been processed\b",
    r"\byour refund has been issued\b",
    r"\bwe have already refunded\b",
    r"\bwe have initiated\b",
    r"\border\s*#\s*\d+\b",  # Fake order numbers
]

# Phrases that offer a genuine next step (refund, return, exchange, etc.)
NEXT_STEP_PATTERNS = [
    r"\brefund\b",
    r"\breturn\b",
    r"\bexchange\b",
    r"\breplace\b",
    r"\bmake this right\b",
    r"\bassist\b",
    r"\bhelp arrange\b",
]


def validate_response(email_text: str, review_text: str = "", rating: int = 1) -> dict:
    """Check whether an AI-generated apology email meets all required criteria.

    Rules checked:
    1. Not empty.
    2. Word count <= 130.
    3. Contains 'Customer Care Team' sign-off.
    4. Offers a concrete next step (refund / return / exchange).
    5. Does NOT falsely claim that an action was already completed.
    6. References specific words from the original review (shows personalization).

    Returns a dict with pass/fail for each check and an overall 'passed_all' flag.
    """
    text = email_text.strip()
    words = text.split()
    word_count = len(words)

    # 1. Not empty
    not_empty = word_count > 0 and "[FALLBACK" not in text and "Error:" not in text

    # 2. Word count
    word_limit_ok = 0 < word_count <= 130

    # 3. Sign-off
    has_signoff = "customer care team" in text.lower()

    # 4. Next step
    has_next_step = any(re.search(p, text, re.IGNORECASE) for p in NEXT_STEP_PATTERNS)

    # 5. No fabricated actions
    fabricated = [
        re.search(p, text, re.IGNORECASE).group(0)
        for p in FABRICATED_ACTION_PATTERNS
        if re.search(p, text, re.IGNORECASE)
    ]
    no_fabrication = len(fabricated) == 0

    # 6. Specificity — does the email share any meaningful words with the review?
    review_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", review_text.lower()))
    common_filler = {"that", "with", "have", "this", "from", "they", "will", "would", "your", "been"}
    review_keywords -= common_filler
    email_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()))
    overlap = review_keywords & email_keywords
    is_specific = len(overlap) >= 2 or len(review_text) < 50

    passed_all = (
        not_empty and word_limit_ok and has_signoff
        and has_next_step and no_fabrication and is_specific
    )

    return {
        "passed_all": passed_all,
        "not_empty": not_empty,
        "word_count": word_count,
        "word_limit_ok": word_limit_ok,
        "has_signoff": has_signoff,
        "has_next_step": has_next_step,
        "no_fabrication": no_fabrication,
        "is_specific": is_specific,
        "fabricated_phrases": fabricated,
        "overlapping_keywords": list(overlap)[:5],
    }
