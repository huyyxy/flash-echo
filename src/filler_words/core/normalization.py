import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()
