"""Configuration settings for the Customer Feedback project."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Womens Clothing E-Commerce Reviews.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Required columns for this dataset
REQUIRED_COLUMNS = ["reviewText", "overall", "summary", "asin", "helpful"]

# Minimum word length for keyword analysis
MIN_WORD_LEN = 3

# Complaint keywords to track
COMPLAINT_TERMS = [
    "small", "large", "cheap", "poor", "returned",
    "refund", "itchy", "see-through", "fit", "color", "fabric",
]

# Stopwords to skip in keyword frequency analysis
CUSTOM_STOPWORDS = {
    "the", "and", "to", "of", "a", "i", "in", "it", "is", "this", "that",
    "was", "for", "with", "my", "on", "but", "have", "not", "so", "dress",
    "be", "are", "at", "as", "or", "from", "by", "an", "they", "you", "we",
    "all", "if", "me", "one", "had", "just", "very", "would", "like", "size",
    "top", "shirt", "wear", "ordered", "bought", "look", "looks", "out",
    "up", "when", "about", "more", "much", "them", "can", "will", "do",
    "get", "got", "really", "even", "which", "there", "were", "been", "am"
}

# Gemini system prompt
SYSTEM_PROMPT = (
    "You are a Customer Support Agent for a retail clothing company. "
    "Write a short (under 130 words), warm, personalized, empathetic apology email. "
    "Reference the specific complaints in the review, avoid generic wording, "
    "offer a concrete next step (refund, exchange or free return), and sign off as "
    "'Customer Care Team'. Do not invent order numbers, customer information, "
    "completed actions, or company policies. Never claim that an action (such as "
    "issuing a refund or generating a shipping label) has already been processed or initiated; "
    "instead offer to arrange it once the customer replies."
)
