"""Gemini GenAI integration service for generating personalized apology emails."""

import os
import re
import time
from typing import Optional, Tuple
from dotenv import load_dotenv

from src.config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT
from src.schemas import AIResponseDraft
from src.validators import validate_apology_response

load_dotenv()


def extract_subject_and_body(text: str) -> Tuple[str, str]:
    """Parse subject line if provided by the model or construct an appropriate default."""
    subject = "Regarding your recent experience with our apparel"
    body = text.strip()

    subject_match = re.search(r"^(?:Subject|Re):\s*(.+)$", body, re.MULTILINE | re.IGNORECASE)
    if subject_match:
        subject = subject_match.group(1).strip()
        # Remove the Subject line from the body
        body = re.sub(r"^(?:Subject|Re):\s*.+\n+", "", body, flags=re.MULTILINE | re.IGNORECASE).strip()

    return subject, body


def generate_apology_email(
    review_text: str,
    rating: int,
    title: Optional[str] = "",
    clothing_id: Optional[int] = None,
    department: Optional[str] = None,
) -> AIResponseDraft:
    """Generate an empathetic, complaint-specific apology email draft using Google Gemini.

    Ensures compliance with:
      - Max 130 words
      - Reference to specific customer complaints
      - Offer of concrete next step (refund / exchange / return)
      - Sign-off: 'Customer Care Team'
      - Zero fabricated actions (never claims return/refund was already completed)
      - No fabricated order numbers or policies
    """
    # Allow environment override or fallback to config
    if "GEMINI_API_KEY" in os.environ:
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        api_key = GEMINI_API_KEY

    model_name = os.getenv("GEMINI_MODEL", GEMINI_MODEL)

    # Check for missing API credentials
    if not api_key or api_key == "your_key_here":
        err_msg = "Gemini API key is not configured in .env. Real AI generation unavailable."
        fallback_body = (
            "Dear Valued Customer,\n\n"
            "Thank you for sharing your candid feedback. We sincerely apologize that your purchase "
            "did not meet your expectations. We would be happy to help arrange a return or refund for you. "
            "Please reply to this email and our team will assist you with the next steps.\n\n"
            "Warm regards,\nCustomer Care Team"
        )
        val = validate_apology_response(fallback_body, review_text, rating)
        return AIResponseDraft(
            subject="Regarding your recent order",
            email_body=fallback_body,
            model_used="None (Fallback)",
            is_fallback=True,
            error_message=err_msg,
            validation=val,
        )

    # Attempt GenAI call with automatic retries for transient spikes
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        title_info = f"Review Title: {title}" if title else "Review Title: (None provided)"
        item_info = f"Department: {department}" if department else ""

        user_content = (
            f"Customer Rating: {rating}/5 Stars\n"
            f"{title_info}\n"
            f"{item_info}\n\n"
            f"Customer Review Text:\n\"{review_text}\""
        )

        # Retry logic for 503 / high demand or 429
        max_retries = 3
        last_error = None
        raw_text = None

        # Candidate models to try in sequence if specified model has 404 restriction
        models_to_try = [model_name]
        if model_name != "gemini-3.1-flash-lite" and model_name != "gemini-3.8-flash":
            models_to_try.extend(["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"])

        for current_model in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=current_model,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=500,
                        ),
                    )
                    raw_text = response.text.strip()
                    model_name = current_model
                    break
                except Exception as exc:
                    err_str = str(exc)
                    last_error = exc
                    # If 404 (model not found / restricted), break inner loop to try fallback candidate
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        break
                    # If 503 (high demand) or 429, wait and retry
                    if attempt < max_retries - 1 and ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str):
                        time.sleep(2 * (attempt + 1))
                        continue
                    break

            if raw_text is not None:
                break

        if raw_text is None:
            raise last_error or RuntimeError("Failed to generate response from Gemini.")

        subject, body = extract_subject_and_body(raw_text)
        val = validate_apology_response(body, review_text, rating)

        return AIResponseDraft(
            subject=subject,
            email_body=body,
            model_used=model_name,
            is_fallback=False,
            error_message=None,
            validation=val,
        )

    except Exception as exc:
        err_msg = f"Gemini API generation failed ({type(exc).__name__}): {exc}"
        fallback_body = (
            "Dear Valued Customer,\n\n"
            "Thank you for reaching out and sharing your feedback. We are truly sorry that your "
            "purchase did not meet our usual quality standards. We would be glad to help arrange "
            "a return, exchange, or refund for you. Please reply to this email so we can take care of this for you.\n\n"
            "Warm regards,\nCustomer Care Team"
        )
        val = validate_apology_response(fallback_body, review_text, rating)
        return AIResponseDraft(
            subject="Regarding your recent order",
            email_body=fallback_body,
            model_used=f"{model_name} (Failed)",
            is_fallback=True,
            error_message=err_msg,
            validation=val,
        )
