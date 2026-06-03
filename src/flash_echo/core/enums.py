from enum import Enum


class Route(str, Enum):
    STATIC_REPLY = "static_reply"
    MINIMIND3_FILLER = "minimind3_filler"
    FALLBACK_FILLER = "fallback_filler"


class ReplyKind(str, Enum):
    COMPLETE_STATIC_REPLY = "complete_static_reply"
    FILLER_PREFIX = "filler_prefix"


class FallbackReason(str, Enum):
    TIMEOUT = "timeout"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_LENGTH = "invalid_length"
    INVALID_BOUNDARY = "invalid_boundary"
    DIRECT_ANSWER = "direct_answer"
    UNSAFE_OUTPUT = "unsafe_output"


class StaticRuleMatchType(str, Enum):
    """静态规则 match.type 取值。"""

    EXACT = "exact"
    PREFIX = "prefix"
    REGEX = "regex"
    EXACT_WITH_TRAILING_PUNCTUATION = "exact_with_trailing_punctuation"
