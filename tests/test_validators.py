"""Tests for response safety, word limit, and anti-fabrication validators."""

import pytest
from src.validators import validate_apology_response


def test_validator_passes_compliant_response():
    email = (
        "Dear Valued Customer,\n\n"
        "Thank you for sharing your feedback. I am so sorry that the fabric and fit of your dress "
        "did not meet your expectations. We pride ourselves on delivering quality apparel, and I apologize "
        "for letting you down. We would be happy to help arrange a return or refund for you. "
        "Please reply to this message and our team will guide you through the process.\n\n"
        "Warm regards,\nCustomer Care Team"
    )
    original_review = "The dress was very poorly made and the fabric felt cheap and see through."

    val = validate_apology_response(email, original_review, rating=1)

    assert val.word_limit is True
    assert val.word_count < 130
    assert val.signoff is True
    assert val.next_step is True
    assert val.no_fabricated_action is True
    assert val.passed_all is True


def test_validator_fails_fabricated_action():
    # Fabricating a completed refund or return label
    email_with_fake_action = (
        "Dear Customer,\n\n"
        "I am so sorry about your dress. I have initiated your refund and generated your return label. "
        "Your order #98765 has been refunded.\n\n"
        "Customer Care Team"
    )
    val = validate_apology_response(email_with_fake_action, "The dress had terrible seams.", rating=1)

    assert val.no_fabricated_action is False
    assert val.passed_all is False
    assert "Detected false completed action claim" in val.details["no_fabricated_action"]


def test_validator_fails_missing_signoff():
    email_without_signoff = (
        "Dear Customer,\n\n"
        "We are sorry your dress was small. We would be happy to arrange a refund for you.\n\n"
        "Sincerely,\nSupport Department"
    )
    val = validate_apology_response(email_without_signoff, "Too small.", rating=1)

    assert val.signoff is False
    assert val.passed_all is False


def test_validator_fails_word_count_exceeded():
    long_email = "word " * 140 + " Customer Care Team refund"
    val = validate_apology_response(long_email, "Some review", rating=1)

    assert val.word_limit is False
    assert val.word_count > 130
    assert val.passed_all is False
