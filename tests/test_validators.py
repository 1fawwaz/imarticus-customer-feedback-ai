"""Tests for the AI response validation logic."""

from src.validators import validate_response

GOOD_EMAIL = (
    "Dear Customer,\n\n"
    "We are truly sorry that the dress was completely see-through and did not fit as expected. "
    "We understand how frustrating this must be and would be glad to help arrange a full refund or exchange. "
    "Please reply to this email so we can assist you right away.\n\n"
    "Sincerely,\nCustomer Care Team"
)

REVIEW = "The fabric was completely see-through and the fit was awful."


def test_good_email_passes_all():
    result = validate_response(GOOD_EMAIL, REVIEW, rating=1)
    assert result["passed_all"] is True


def test_word_count_under_130():
    result = validate_response(GOOD_EMAIL, REVIEW)
    assert result["word_count"] <= 130
    assert result["word_limit_ok"] is True


def test_has_signoff():
    result = validate_response(GOOD_EMAIL, REVIEW)
    assert result["has_signoff"] is True


def test_has_next_step():
    result = validate_response(GOOD_EMAIL, REVIEW)
    assert result["has_next_step"] is True


def test_no_fabrication():
    result = validate_response(GOOD_EMAIL, REVIEW)
    assert result["no_fabrication"] is True


def test_is_specific():
    result = validate_response(GOOD_EMAIL, REVIEW)
    assert result["is_specific"] is True


def test_empty_email_fails():
    result = validate_response("", REVIEW)
    assert result["not_empty"] is False
    assert result["passed_all"] is False


def test_too_long_email_fails():
    long_email = "word " * 200
    result = validate_response(long_email, REVIEW)
    assert result["word_limit_ok"] is False


def test_missing_signoff_fails():
    email = (
        "Dear Customer, we are sorry about your experience. "
        "We would be happy to arrange a refund. Please reply to us. "
        "Best regards, Support Staff"
    )
    result = validate_response(email, REVIEW)
    assert result["has_signoff"] is False


def test_fabricated_action_detected():
    bad_email = (
        "Dear Customer, we have already issued a refund for your order. "
        "Your refund has been processed and sent. "
        "Customer Care Team"
    )
    result = validate_response(bad_email, REVIEW)
    assert result["no_fabrication"] is False
    assert len(result["fabricated_phrases"]) > 0


def test_missing_next_step_fails():
    email = (
        "Dear Customer, we are sorry for your experience. "
        "We appreciate your feedback. "
        "Customer Care Team"
    )
    result = validate_response(email, REVIEW)
    assert result["has_next_step"] is False


def test_result_has_all_keys():
    result = validate_response(GOOD_EMAIL, REVIEW)
    for key in ["passed_all", "not_empty", "word_count", "word_limit_ok",
                "has_signoff", "has_next_step", "no_fabrication", "is_specific"]:
        assert key in result
