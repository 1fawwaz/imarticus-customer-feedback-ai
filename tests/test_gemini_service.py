"""Tests for Gemini service integration and graceful error handling."""

import os
import pytest
from src.gemini_service import generate_apology_email, extract_subject_and_body


def test_extract_subject_and_body():
    sample = "Subject: Regarding your recent order\n\nDear Customer,\nWe are very sorry."
    subject, body = extract_subject_and_body(sample)
    assert subject == "Regarding your recent order"
    assert "Subject:" not in body
    assert body.startswith("Dear Customer,")


def test_gemini_service_handles_missing_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    draft = generate_apology_email(
        review_text="This dress had awful fabric and was see through.",
        rating=1,
        title="Terrible dress"
    )

    assert draft.is_fallback is True
    assert "not configured" in draft.error_message
    assert draft.validation.signoff is True
    assert draft.validation.word_limit is True
    assert draft.validation.no_fabricated_action is True
