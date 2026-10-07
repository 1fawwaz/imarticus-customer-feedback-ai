"""Response validation module for checking GenAI apology drafts."""

import re
from typing import Dict, Any, List
from src.schemas import ValidationResult


# Disallowed phrases that falsely claim an action has already been performed in the real world
FABRICATED_ACTION_PATTERNS = [
    r"\bi have initiated\b",
    r"\bi have processed\b",
    r"\bi have issued\b",
    r"\bi have created a\b",
    r"\bi have refunded\b",
    r"\bi have generated\b",
    r"\byour refund has been processed\b",
    r"\byour refund has been issued\b",
    r"\bwe have already refunded\b",
    r"\bwe have already processed\b",
    r"\bwe have initiated\b",
    r"\border\s*#\s*\d+\b",  # Fake order numbers
]

# Patterns representing genuine concrete offers for assistance / next steps
NEXT_STEP_PATTERNS = [
    r"\brefund\b",
    r"\breturn\b",
    r"\bexchange\b",
    r"\breplace\b",
    r"\bmake this right\b",
    r"\bassist\b",
    r"\bhelp arrange\b",
]


def validate_apology_response(
    email_text: str,
    original_review_text: str = "",
    rating: int = 1,
) -> ValidationResult:
    """Validate customer apology email draft against strict safety and quality criteria.

    Criteria:
      1. Word count <= 130.
      2. Sign-off contains 'Customer Care Team'.
      3. Contains concrete next step (refund, return, exchange).
      4. Does NOT falsely claim actions have already occurred or invent order numbers.
      5. Demonstrates context-specificity to the review.
      6. Is non-empty and not an unhandled system failure.
    """
    clean_text = email_text.strip()
    words = clean_text.split()
    word_count = len(words)
    details: Dict[str, str] = {}

    # Check 1: Not empty & not failure indicator
    is_empty_or_error = (
        word_count == 0 or
        "[FALLBACK" in clean_text or
        "Error:" in clean_text or
        "AI generation failed" in clean_text
    )
    not_empty = not is_empty_or_error
    details["not_empty"] = "Valid non-empty response" if not_empty else "Response is empty or contains an error message."

    # Check 2: Word limit <= 130
    word_limit = 0 < word_count <= 130
    details["word_limit"] = f"{word_count} words (Limit: <= 130)"

    # Check 3: Required sign-off
    has_signoff = "customer care team" in clean_text.lower()
    details["signoff"] = "Contains 'Customer Care Team'" if has_signoff else "Missing required 'Customer Care Team' sign-off."

    # Check 4: Concrete next step included
    has_next_step = any(re.search(pat, clean_text, re.IGNORECASE) for pat in NEXT_STEP_PATTERNS)
    details["next_step"] = "Concrete next step included" if has_next_step else "Missing actionable next step (refund/return/exchange)."

    # Check 5: Anti-fabrication (No claims of already-completed actions or fake order numbers)
    fabricated_detected = []
    for pat in FABRICATED_ACTION_PATTERNS:
        match = re.search(pat, clean_text, re.IGNORECASE)
        if match:
            fabricated_detected.append(match.group(0))

    no_fabricated_action = len(fabricated_detected) == 0
    if no_fabricated_action:
        details["no_fabricated_action"] = "No fabricated actions or fake order numbers detected."
    else:
        details["no_fabricated_action"] = f"Detected false completed action claim: '{', '.join(fabricated_detected)}'"

    # Check 6: Specificity / relevance to review context
    # Extract significant words (len >= 4) from original review
    review_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", original_review_text.lower()))
    # Remove ultra-common email words
    common_filler = {"that", "with", "have", "this", "from", "they", "will", "would", "about", "there", "what", "your", "been", "were"}
    review_keywords = review_keywords - common_filler

    email_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", clean_text.lower()))
    overlap = review_keywords.intersection(email_words)

    # Specific if at least 2 context words overlap or original review is very short
    has_specificity = len(overlap) >= 2 or len(original_review_text) < 50
    details["specificity"] = f"Contextual keywords matched ({len(overlap)}): {list(overlap)[:5]}" if has_specificity else "Generic wording detected; lacks specific reference to customer complaint."

    passed_all = (
        not_empty and
        word_limit and
        has_signoff and
        has_next_step and
        no_fabricated_action and
        has_specificity
    )

    return ValidationResult(
        word_limit=word_limit,
        word_count=word_count,
        signoff=has_signoff,
        specificity=has_specificity,
        next_step=has_next_step,
        no_fabricated_action=no_fabricated_action,
        not_empty=not_empty,
        passed_all=passed_all,
        details=details,
    )
