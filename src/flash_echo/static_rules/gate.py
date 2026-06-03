from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash_echo.core.enums import StaticRuleMatchType
from flash_echo.core.normalization import normalize_query

_TRAILING_NON_QUESTION_PUNCT_RE = re.compile(r"[。！!，,、；;：:~～…]+$")


@dataclass(frozen=True)
class StaticRuleMatch:
    rule_id: str
    text: str


@dataclass(frozen=True)
class StaticRule:
    rule_id: str
    match_type: StaticRuleMatchType
    patterns: tuple[str, ...]
    normalized_patterns: tuple[str, ...]
    regex_patterns: tuple[re.Pattern[str], ...]
    negative_examples: tuple[str, ...]
    replies: dict[str, list[str]]
    enabled: bool = True


def parse_match_type(raw: str | None) -> StaticRuleMatchType | None:
    if not raw:
        return StaticRuleMatchType.EXACT
    try:
        return StaticRuleMatchType(raw)
    except ValueError:
        return None


def strip_non_question_trailing_punctuation(text: str) -> str:
    return _TRAILING_NON_QUESTION_PUNCT_RE.sub("", text).strip()


class StaticReplyGate:
    def __init__(self, rules: list[StaticRule], *, version: str) -> None:
        self.version = version
        self._rules = [rule for rule in rules if rule.enabled]

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> StaticReplyGate:
        version = str(payload.get("version", "static-rules-v1"))
        raw_rules = payload.get("rules", [])
        rules: list[StaticRule] = []
        if isinstance(raw_rules, list):
            for item in raw_rules:
                if not isinstance(item, dict) or not item.get("enabled", True):
                    continue
                match_cfg = item.get("match", {})
                if not isinstance(match_cfg, dict):
                    continue
                match_type = parse_match_type(match_cfg.get("type"))
                raw_patterns = match_cfg.get("patterns", [])
                patterns = tuple(raw_patterns) if isinstance(raw_patterns, list) else ()
                replies = item.get("replies", {})
                if match_type is None or not patterns or not isinstance(replies, dict):
                    continue
                normalized_replies = {
                    persona: list(candidates)
                    for persona, candidates in replies.items()
                    if isinstance(candidates, list) and candidates
                }
                normalized_patterns = tuple(normalize_query(pattern) for pattern in patterns)
                regex_patterns = (
                    tuple(re.compile(pattern) for pattern in patterns)
                    if match_type is StaticRuleMatchType.REGEX
                    else ()
                )
                rules.append(
                    StaticRule(
                        rule_id=str(item["rule_id"]),
                        match_type=match_type,
                        patterns=patterns,
                        normalized_patterns=normalized_patterns,
                        regex_patterns=regex_patterns,
                        negative_examples=tuple(item.get("negative_examples", [])),
                        replies=normalized_replies,
                        enabled=True,
                    )
                )
        return cls(rules, version=version)

    @classmethod
    def from_file(cls, path: Path) -> StaticReplyGate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_config(payload)

    def match(self, query: str, *, persona_tag: str) -> StaticRuleMatch | None:
        normalized = normalize_query(query)
        for rule in self._rules:
            if not self._matches_rule(normalized, rule):
                continue
            reply = self._choose_reply(rule, persona_tag)
            if reply:
                return StaticRuleMatch(rule_id=rule.rule_id, text=reply)
        return None

    @staticmethod
    def _matches_rule(normalized_query: str, rule: StaticRule) -> bool:
        for negative in rule.negative_examples:
            if normalize_query(negative) == normalized_query:
                return False
        return StaticReplyGate._matches_patterns(normalized_query, rule)

    @staticmethod
    def _matches_patterns(normalized_query: str, rule: StaticRule) -> bool:
        match rule.match_type:
            case StaticRuleMatchType.EXACT:
                return normalized_query in rule.normalized_patterns
            case StaticRuleMatchType.EXACT_WITH_TRAILING_PUNCTUATION:
                stripped_query = strip_non_question_trailing_punctuation(normalized_query)
                return (
                    normalized_query in rule.normalized_patterns
                    or stripped_query in rule.normalized_patterns
                )
            case StaticRuleMatchType.PREFIX:
                return any(
                    normalized_query.startswith(pattern) for pattern in rule.normalized_patterns
                )
            case StaticRuleMatchType.REGEX:
                return any(
                    compiled.fullmatch(normalized_query) for compiled in rule.regex_patterns
                )
        return False

    @staticmethod
    def _choose_reply(rule: StaticRule, persona_tag: str) -> str | None:
        candidates = rule.replies.get(persona_tag) or rule.replies.get("default")
        if not candidates:
            return None
        return random.choice(candidates)


DEFAULT_STATIC_RULES: dict[str, Any] = {
    "version": "static-rules-v1",
    "locale": "zh-CN",
    "rules": [
        {
            "rule_id": "greeting.reply.001",
            "category": "greeting",
            "match": {
                "type": "exact_with_trailing_punctuation",
                "patterns": ["你好", "您好", "哈喽", "嗨"],
            },
            "negative_examples": [],
            "replies": {
                "female_receptionist": ["你好！", "您好！"],
                "male_white_collar": ["你好。", "您好。"],
                "default": ["你好！"],
            },
            "enabled": True,
        },
        {
            "rule_id": "thanks.reply.001",
            "category": "thanks",
            "match": {
                "type": "exact_with_trailing_punctuation",
                "patterns": ["谢谢", "多谢", "辛苦了"],
            },
            "negative_examples": ["谢谢你帮我分析一下这个合同"],
            "replies": {
                "female_receptionist": ["不客气！", "应该的！"],
                "male_white_collar": ["不客气。"],
                "default": ["不客气！"],
            },
            "enabled": True,
        },
        {
            "rule_id": "confirm.reply.001",
            "category": "confirm",
            "match": {
                "type": "exact_with_trailing_punctuation",
                "patterns": ["好的", "好", "嗯", "行", "可以"],
            },
            "negative_examples": ["好的，那你帮我查一下"],
            "replies": {
                "female_receptionist": ["好的！", "没问题！"],
                "male_white_collar": ["好的。", "明白。"],
                "default": ["好的！"],
            },
            "enabled": True,
        },
        {
            "rule_id": "farewell.reply.001",
            "category": "farewell",
            "match": {
                "type": "exact_with_trailing_punctuation",
                "patterns": ["再见", "拜拜", "下次聊"],
            },
            "negative_examples": [],
            "replies": {
                "female_receptionist": ["再见！", "下次见！"],
                "male_white_collar": ["再见。", "下次聊。"],
                "default": ["再见！"],
            },
            "enabled": True,
        },
    ],
}
