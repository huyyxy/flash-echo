from enum import Enum


class FillerType(str, Enum):
    NONE = "NONE"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    THINKING = "THINKING"
    FRAME = "FRAME"
    EMPATHY = "EMPATHY"
    RETRIEVAL = "RETRIEVAL"
    CLARIFY_LEADIN = "CLARIFY_LEADIN"


FILLER_PRIORITY: tuple[FillerType, ...] = (
    FillerType.EMPATHY,
    FillerType.CLARIFY_LEADIN,
    FillerType.RETRIEVAL,
    FillerType.ACKNOWLEDGE,
    FillerType.FRAME,
    FillerType.THINKING,
    FillerType.NONE,
)

TRIGGERED_TYPES = frozenset(t for t in FillerType if t is not FillerType.NONE)


def trigger_for_type(filler_type: FillerType) -> int:
    return 0 if filler_type is FillerType.NONE else 1
