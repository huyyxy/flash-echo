import re
import unicodedata

NORMALIZER_VERSION = "normalizer-v1"
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()
