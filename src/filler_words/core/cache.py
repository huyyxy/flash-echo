from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic

from filler_words.core.labels import FillerType


@dataclass(frozen=True)
class ClassificationResult:
    filler_type: FillerType
    confidence: float | None = None

    @property
    def trigger(self) -> int:
        return 0 if self.filler_type is FillerType.NONE else 1


@dataclass(frozen=True)
class CacheEntry:
    value: ClassificationResult
    expires_at: float


class LruTtlCache:
    def __init__(self, max_entries: int = 10_000, ttl_seconds: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> ClassificationResult | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= monotonic():
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return entry.value

    def set(self, key: str, value: ClassificationResult) -> None:
        self._entries[key] = CacheEntry(value=value, expires_at=monotonic() + self.ttl_seconds)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
