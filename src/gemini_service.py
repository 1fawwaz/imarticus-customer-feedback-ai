"""Generate AI apology email drafts using Google Gemini."""

import os
import re
import time
from typing import Optional, Tuple
from dotenv import load_dotenv

from src.config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT
from src.validators import validate_response

load_dotenv()


def _parse_subject_and_body(text: str) -> Tuple[str, str]:
    """Extract the subject line if the model included one, otherwise use a default."""
    subject = "Regarding your recent experience with our apparel"
    body = text.strip()

    match = re.search(r"^(?:Subject|Re):\s*(.+)$", body, re.MULTILINE | re.IGNORECASE)
    if match:
        subject = match.group(1).strip()
        body = re.sub(r"^(?:Subject|Re):\s*.+\n+", "", body, flags=re.MULTILINE | re.IGNORECASE).strip()

    return subject, body


def generate_apology_email(
    review_text: str,
    rating: int,
    summary: Optional[str] = "",
    asin: Optional[str] = None,
) -> dict:
    """Use Google Gemini to draft a personalized apology email for a critical review.

    The email must:
    - Be under 130 words
    - Reference the customer's specific complaint
    - Offer a next step (refund, return, or exchange)
    - Sign off as 'Customer Care Team'
    - NOT claim any action has already been completed

    Returns a dict with keys: subject, email_body, model_used, is_fallback, error, validation.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    model_name = os.getenv("GEMINI_MODEL", GEMINI_MODEL)

    # Fallback if no API key
    if not api_key or api_key == "your_key_here":
        fallback_body = (
            "Dear Valued Customer,\n\n"
            "Thank you for sharing your feedback. We sincerely apologize that your purchase "
            "did not meet your expectations. We would be happy to help arrange a return or refund. "
            "Please reply to this email and our team will assist you with the next steps.\n\n"
            "Warm regards,\nCustomer Care Team"
        )
        return {
            "subject": "Regarding your recent order",
            "email_body": fallback_body,
            "model_used": "None (API key not set)",
            "is_fallback": True,
            "error": "GEMINI_API_KEY is not configured.",
            "validation": validate_response(fallback_body, review_text, rating),
        }

    # Build the prompt
    content_parts = [f"Customer Rating: {rating}/5 Stars"]
    if summary:
        content_parts.append(f"Review Summary: {summary}")
    if asin:
        content_parts.append(f"Product ID: {asin}")
    content_parts.append(f"\nCustomer Review:\n\"{review_text}\"")
    user_content = "\n".join(content_parts)

    # Try the configured model, fall back to other Gemini variants if needed
    models_to_try = [model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    # Remove duplicates while preserving order
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    raw_text = None
    last_error = None
    used_model = model_name

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        for current_model in models_to_try:
            for attempt in range(3):
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
                    used_model = current_model
                    break
                except Exception as exc:
                    err_str = str(exc)
                    last_error = exc
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        break  # This model not available, try next
                    if attempt < 2 and ("503" in err_str or "429" in err_str):
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
            if raw_text is not None:
                break

        if raw_text is None:
            raise last_error or RuntimeError("No response received from Gemini.")

        subject, body = _parse_subject_and_body(raw_text)
        return {
            "subject": subject,
            "email_body": body,
            "model_used": used_model,
            "is_fallback": False,
            "error": None,
            "validation": validate_response(body, review_text, rating),
        }

    except Exception as exc:
        fallback_body = (
            "Dear Valued Customer,\n\n"
            "Thank you for reaching out. We are truly sorry that your purchase did not meet "
            "our usual quality standards. We would be glad to help arrange a return, exchange, "
            "or refund. Please reply so we can take care of this for you.\n\n"
            "Warm regards,\nCustomer Care Team"
        )
        return {
            "subject": "Regarding your recent order",
            "email_body": fallback_body,
            "model_used": f"{used_model} (failed)",
            "is_fallback": True,
            "error": f"{type(exc).__name__}: {exc}",
            "validation": validate_response(fallback_body, review_text, rating),
        }
