from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from filler_words.api.schemas import ChatCompletionRequest, RequestMetadata
from filler_words.app import build_service, create_app
from filler_words.core.cache import LruTtlCache
from filler_words.core.enums import FallbackReason, ReplyKind, Route
from filler_words.fallback.registry import FallbackPrefixRegistry
from filler_words.inference.minimind3_onnx import GenerationResult
from filler_words.inference.persona_router import PersonaModelRouter, PersonaRoute
from filler_words.service.filler_service import FillerReplyService, ServiceSettings
from filler_words.static_rules.gate import DEFAULT_STATIC_RULES, StaticReplyGate


@dataclass
class FakeGenerator:
    text: str
    timed_out: bool = False
    prompt_tokens: int = 12
    completion_tokens: int = 6

    model_version: str = "minimind3-filler-male-white-collar-v1.0.0"

    def generate(self, query: str) -> GenerationResult:
        return GenerationResult(
            text=self.text,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            timed_out=self.timed_out,
        )

    @property
    def validator(self):
        from filler_words.inference.validator import OutputValidator, ValidationConfig

        return OutputValidator(
            ValidationConfig(
                min_chars=3,
                max_chars=48,
                require_punctuation_endings=("，", "。", "！"),
                forbid_answer_patterns=(),
            )
        )


class FakeModelPool:
    def __init__(self, generator: FakeGenerator) -> None:
        self.generator = generator

    def get(self, bundle_dir: Path) -> FakeGenerator:
        return self.generator


def _build_test_service(
    *,
    generator: FakeGenerator | None = None,
    persona_tag: str = "male_white_collar",
    tmp_path: Path | None = None,
) -> FillerReplyService:
    deploy_root = tmp_path or Path("/tmp/flash-echo-test/deploy")
    bundle_dir = deploy_root / "minimind3-filler-male-white-collar-v1.0.0"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.onnx").touch()
    router = PersonaModelRouter(
        {
            persona_tag: PersonaRoute(
                persona_tag=persona_tag,
                model_version="minimind3-filler-male-white-collar-v1.0.0",
                bundle_dir=bundle_dir,
                enabled=True,
            )
        },
        version="model-router-test",
    )
    return FillerReplyService(
        static_gate=StaticReplyGate.from_config(DEFAULT_STATIC_RULES),
        persona_router=router,
        fallback_registry=FallbackPrefixRegistry.from_config(
            {
                "version": "fallback-prefix-v1",
                "zh-CN": {"male_white_collar": ["我想一下，"], "default": ["我想一下，"]},
            }
        ),
        model_pool=FakeModelPool(generator or FakeGenerator("这个问题可以这样看，")),
        cache=LruTtlCache(max_entries=100, ttl_seconds=60),
        settings=ServiceSettings(),
    )


def test_static_reply_for_greeting() -> None:
    service = _build_test_service()

    response = service.create_completion(
        ChatCompletionRequest(
            model="filler-reply-minimind3",
            messages=[{"role": "user", "content": "你好"}],
            metadata=RequestMetadata(persona_tag="male_white_collar", request_id="req-1"),
        )
    )

    assert response.choices[0].message.content in {"你好。", "您好。"}
    assert response.filler_reply.route is Route.STATIC_REPLY
    assert response.filler_reply.reply_kind is ReplyKind.COMPLETE_STATIC_REPLY
    assert response.filler_reply.continue_with_main_llm is False
    assert response.usage.total_tokens == 0


def test_minimind3_route_returns_prefix() -> None:
    service = _build_test_service()

    response = service.create_completion(
        ChatCompletionRequest(
            model="filler-reply-minimind3",
            messages=[{"role": "user", "content": "你怎么看 AI 对教育行业的影响？"}],
            metadata=RequestMetadata(persona_tag="male_white_collar", request_id="req-2"),
        )
    )

    assert response.choices[0].message.content == "这个问题可以这样看，"
    assert response.filler_reply.route is Route.MINIMIND3_FILLER
    assert response.filler_reply.continue_with_main_llm is True
    assert response.filler_reply.model_version == "minimind3-filler-male-white-collar-v1.0.0"


def test_minimind3_route_accepts_ascii_punctuation_boundary() -> None:
    service = _build_test_service(generator=FakeGenerator("关于AI对教育的影响,"))

    response = service.create_completion(
        ChatCompletionRequest(
            model="filler-reply-minimind3",
            messages=[{"role": "user", "content": "你怎么看 AI 对教育行业的影响？"}],
            metadata=RequestMetadata(persona_tag="male_white_collar", request_id="req-ascii"),
        )
    )

    assert response.choices[0].message.content == "关于AI对教育的影响,"
    assert response.filler_reply.route is Route.MINIMIND3_FILLER
    assert response.filler_reply.fallback_reason is None


def test_unknown_persona_uses_fallback() -> None:
    service = _build_test_service()

    response = service.create_completion(
        ChatCompletionRequest(
            model="filler-reply-minimind3",
            messages=[{"role": "user", "content": "帮我写一封请假邮件"}],
            metadata=RequestMetadata(persona_tag="unknown_persona", request_id="req-3"),
        )
    )

    assert response.filler_reply.route is Route.FALLBACK_FILLER
    assert response.filler_reply.fallback_reason is FallbackReason.MODEL_UNAVAILABLE


def test_invalid_model_output_uses_fallback() -> None:
    service = _build_test_service(generator=FakeGenerator("苹果英文是 apple"))

    response = service.create_completion(
        ChatCompletionRequest(
            model="filler-reply-minimind3",
            messages=[{"role": "user", "content": "苹果英文怎么说"}],
            metadata=RequestMetadata(persona_tag="male_white_collar"),
        )
    )

    assert response.filler_reply.route is Route.FALLBACK_FILLER
    assert response.filler_reply.fallback_reason is FallbackReason.DIRECT_ANSWER


def test_http_chat_completions_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "filler-reply-minimind3",
            "messages": [{"role": "user", "content": "谢谢"}],
            "metadata": {"persona_tag": "male_white_collar", "request_id": "req-http"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "不客气。"
    assert payload["filler_reply"]["route"] == "static_reply"


def test_stream_request_returns_501() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "filler-reply-minimind3",
            "stream": True,
            "messages": [{"role": "user", "content": "你好"}],
            "metadata": {"persona_tag": "male_white_collar"},
        },
    )

    assert response.status_code == 501
    assert response.json()["error"]["code"] == "stream_not_supported"
