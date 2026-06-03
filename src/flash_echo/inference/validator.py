from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash_echo.core.enums import FallbackReason


@dataclass(frozen=True)
class ValidationConfig:
    min_chars: int
    max_chars: int
    require_punctuation_endings: tuple[str, ...]
    forbid_answer_patterns: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: FallbackReason | None = None


_DIRECT_ANSWER_PATTERNS = (
    re.compile(r"答案是"),
    re.compile(r"英文是\s*\w"),
    re.compile(r"等于\s*\d"),
    re.compile(r"我查到了"),
    re.compile(r"已经帮你"),
    re.compile(r"我已经"),
)

_ASCII_PUNCTUATION_EQUIVALENTS = {
    "。": ".",
    "，": ",",
    "！": "!",
    "？": "?",
    "、": ",",
}


class OutputValidator:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = ValidationConfig(
            min_chars=config.min_chars,
            max_chars=config.max_chars,
            require_punctuation_endings=_expand_punctuation_endings(
                config.require_punctuation_endings
            ),
            forbid_answer_patterns=config.forbid_answer_patterns,
        )
        compiled = [re.compile(pattern) for pattern in config.forbid_answer_patterns]
        self._forbid_patterns = compiled + list(_DIRECT_ANSWER_PATTERNS)

    @classmethod
    def from_inference_config(cls, payload: dict[str, Any]) -> OutputValidator:
        validation = payload.get("validation", {})
        endings = tuple(validation.get("require_punctuation_endings", ["。", "，", "！", "？", "、"]))
        return cls(
            ValidationConfig(
                min_chars=int(validation.get("min_chars", 3)),
                max_chars=int(validation.get("max_chars", 48)),
                require_punctuation_endings=endings,
                forbid_answer_patterns=tuple(validation.get("forbid_answer_patterns", [])),
            )
        )

    def validate(self, text: str) -> ValidationResult:
        stripped = text.strip()
        if not stripped:
            return ValidationResult(False, FallbackReason.INVALID_LENGTH)

        char_count = len(stripped)
        if char_count < self.config.min_chars or char_count > self.config.max_chars:
            return ValidationResult(False, FallbackReason.INVALID_LENGTH)

        for pattern in self._forbid_patterns:
            if pattern.search(stripped):
                return ValidationResult(False, FallbackReason.DIRECT_ANSWER)

        if self._looks_unsafe(stripped):
            return ValidationResult(False, FallbackReason.UNSAFE_OUTPUT)

        if not any(stripped.endswith(ending) for ending in self.config.require_punctuation_endings):
            return ValidationResult(False, FallbackReason.INVALID_BOUNDARY)

        return ValidationResult(True)

    @staticmethod
    def _looks_unsafe(text: str) -> bool:
        unsafe_terms = ("去死", "滚", "傻逼", "操你", "色情")
        return any(term in text for term in unsafe_terms)


def _expand_punctuation_endings(endings: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for ending in endings:
        if ending not in expanded:
            expanded.append(ending)
        equivalent = _ASCII_PUNCTUATION_EQUIVALENTS.get(ending)
        if equivalent and equivalent not in expanded:
            expanded.append(equivalent)
    return tuple(expanded)


def load_inference_config(bundle_dir: Path) -> dict[str, Any]:
    config_path = bundle_dir / "inference_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing inference_config.json in {bundle_dir}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid inference_config.json in {bundle_dir}")
    return payload
