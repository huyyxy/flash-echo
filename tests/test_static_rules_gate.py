from __future__ import annotations

from flash_echo.core.enums import StaticRuleMatchType
from flash_echo.static_rules.gate import StaticReplyGate, parse_match_type


def _gate_from_rules(rules: list[dict]) -> StaticReplyGate:
    return StaticReplyGate.from_config({"version": "static-rules-test", "rules": rules})


def test_parse_match_type_values() -> None:
    assert parse_match_type("exact") is StaticRuleMatchType.EXACT
    assert parse_match_type("prefix") is StaticRuleMatchType.PREFIX
    assert parse_match_type("regex") is StaticRuleMatchType.REGEX
    assert (
        parse_match_type("exact_with_trailing_punctuation")
        is StaticRuleMatchType.EXACT_WITH_TRAILING_PUNCTUATION
    )
    assert parse_match_type(None) is StaticRuleMatchType.EXACT
    assert parse_match_type("alias") is None
    assert parse_match_type("exact_or_alias") is None
    assert parse_match_type("unknown") is None


def test_exact_match_accepts_multiple_equivalent_patterns() -> None:
    gate = _gate_from_rules(
        [
            {
                "rule_id": "greeting.reply.001",
                "match": {"type": "exact", "patterns": ["你好", "您好"]},
                "negative_examples": [],
                "replies": {"default": ["你好！"]},
                "enabled": True,
            }
        ]
    )

    result = gate.match("您好", persona_tag="default")
    assert result is not None
    assert result.text == "你好！"


def test_exact_match() -> None:
    gate = _gate_from_rules(
        [
            {
                "rule_id": "exact.reply.001",
                "match": {"type": "exact", "patterns": ["收到"]},
                "negative_examples": [],
                "replies": {"default": ["好的。"]},
                "enabled": True,
            }
        ]
    )

    assert gate.match("收到", persona_tag="default") is not None
    assert gate.match("收到了", persona_tag="default") is None


def test_exact_with_trailing_punctuation_match() -> None:
    gate = _gate_from_rules(
        [
            {
                "rule_id": "punctuation.reply.001",
                "match": {
                    "type": "exact_with_trailing_punctuation",
                    "patterns": ["你好", "好的"],
                },
                "negative_examples": [],
                "replies": {"default": ["你好！"]},
                "enabled": True,
            }
        ]
    )

    assert gate.match("你好", persona_tag="default") is not None
    assert gate.match("你好。", persona_tag="default") is not None
    assert gate.match("你好！", persona_tag="default") is not None
    assert gate.match("你好，", persona_tag="default") is not None
    assert gate.match("你好~", persona_tag="default") is not None
    assert gate.match("你好？", persona_tag="default") is None
    assert gate.match("好的，那你帮我查一下", persona_tag="default") is None


def test_prefix_match() -> None:
    gate = _gate_from_rules(
        [
            {
                "rule_id": "prefix.reply.001",
                "match": {"type": "prefix", "patterns": ["稍等"]},
                "negative_examples": [],
                "replies": {"default": ["好的。"]},
                "enabled": True,
            }
        ]
    )

    assert gate.match("稍等一下", persona_tag="default") is not None
    assert gate.match("请稍等", persona_tag="default") is None


def test_regex_match() -> None:
    gate = _gate_from_rules(
        [
            {
                "rule_id": "regex.reply.001",
                "match": {"type": "regex", "patterns": [r"^(ok|OK)$"]},
                "negative_examples": [],
                "replies": {"default": ["好的。"]},
                "enabled": True,
            }
        ]
    )

    assert gate.match("ok", persona_tag="default") is not None
    assert gate.match("OK", persona_tag="default") is not None
    assert gate.match("ok please", persona_tag="default") is None


def test_invalid_match_type_is_skipped() -> None:
    gate = _gate_from_rules(
        [
            {
                "rule_id": "invalid.reply.001",
                "match": {"type": "unsupported", "patterns": ["你好"]},
                "negative_examples": [],
                "replies": {"default": ["你好！"]},
                "enabled": True,
            }
        ]
    )

    assert gate.match("你好", persona_tag="default") is None


def test_unsupported_match_types_are_skipped() -> None:
    gate = _gate_from_rules(
        [
            {
                "rule_id": "unsupported.reply.001",
                "match": {"type": "exact_or_alias", "patterns": ["谢谢"]},
                "negative_examples": [],
                "replies": {"default": ["不客气！"]},
                "enabled": True,
            }
        ]
    )

    assert gate.match("谢谢", persona_tag="default") is None
