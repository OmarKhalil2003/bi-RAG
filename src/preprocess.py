"""
Query and Text Preprocessing Module
Bilingual Document Q&A (RAG) System

Provides canonical normalization for Arabic and English queries and passages:
- Unicode NFKC normalization
- Stripping invisible control characters and directional isolates
- Normalizing Arabic diacritics (Tashkeel) and elongation (Tatweel/Kashida)
- Normalizing Arabic-Indic numerals (٠-٩) to standard digits (0-9)
- Normalizing whitespace and punctuation
"""

import re
import unicodedata
from typing import Dict, Any, List

# Arabic diacritics (Fatha, Damma, Kasra, Sukun, Tanwin, Shadda, etc.)
ARABIC_DIACRITICS_REGEX = re.compile(r"[\u064B-\u065F\u0670]")

# Arabic tatweel (kashida)
ARABIC_TATWEEL_REGEX = re.compile(r"\u0640")

# Arabic-Indic digits mapping
ARABIC_INDIC_DIGITS = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"
}
ARABIC_INDIC_REGEX = re.compile(r"[٠-٩]")


def normalize_arabic_digits(text: str) -> str:
    """Convert Eastern Arabic-Indic numerals (٠-٩) to standard digits (0-9)."""
    return ARABIC_INDIC_REGEX.sub(lambda m: ARABIC_INDIC_DIGITS[m.group(0)], text)


def normalize_query(query: str) -> str:
    """
    Apply canonical normalization to user queries before embedding and retrieval.
    Preserves semantic content while eliminating orthographic and encoding noise.
    """
    if not query:
        return ""

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", query)

    # 2. Remove invisible control characters, null bytes, and directional marks (RLM, LRM)
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200e\u200f\u202a-\u202e\ufeff]", "", normalized)

    # 3. Strip Arabic diacritics (Tashkeel)
    normalized = ARABIC_DIACRITICS_REGEX.sub("", normalized)

    # 4. Strip Arabic tatweel (elongation)
    normalized = ARABIC_TATWEEL_REGEX.sub("", normalized)

    # 5. Convert Arabic-Indic numerals to standard ASCII numerals
    normalized = normalize_arabic_digits(normalized)

    # 6. Ensure clean spacing around punctuation and collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def strip_chunk_header(chunk_text: str) -> str:
    """
    Extract raw passage content by stripping the contextual document header
    e.g. '[Document: ... | Page: ...]' or '[المستند: ... | الصفحة: ...]'.
    """
    header_pattern = r"^\[(Document|المستند):[^\]]+\]\s*"
    return re.sub(header_pattern, "", chunk_text, flags=re.MULTILINE).strip()
