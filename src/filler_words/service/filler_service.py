from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from filler_words.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FillerReplyMeta,
    UsageInfo,
)
from filler_words.core.cache import CachedFillerResult, LruTtlCache
from filler_words.core.enums import FallbackReason, ReplyKind, Route
from filler_words.core.normalization import NORMALIZER_VERSION, normalize_query
from filler_words.fallback.registry import FallbackPrefixRegistry
from filler_words.inference.minimind3_onnx import MiniMind3ModelPool
from filler_words.inference.persona_router import PersonaModelRouter
from filler_words.static_rules.gate import StaticReplyGate


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceSettings:
    service_model_name: str = "filler-reply-minimind3"
    max_query_chars: int = 512


class FillerReplyService:
    def __init__(
        self,
        *,
        static_gate: StaticReplyGate,
        persona_router: PersonaModelRouter,
        fallback_registry: FallbackPrefixRegistry,
        model_pool: MiniMind3ModelPool,
        cache: LruTtlCache,
        settings: ServiceSettings | None = None,
    ) -> None:
        self.static_gate = static_gate
        self.persona_router = persona_router
        self.fallback_registry = fallback_registry
        self.model_pool = model_pool
        self.cache = cache
        self.settings = settings or ServiceSettings()

    def create_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        metadata = request.metadata
        if metadata is None:
            raise ValueError("metadata.persona_tag is required")

        raw_query = request.extract_user_query()
        normalized_query = normalize_query(raw_query)
        if not normalized_query:
            raise ValueError("messages must contain at least one non-empty user message")
        if len(normalized_query) > self.settings.max_query_chars:
            normalized_query = normalized_query[: self.settings.max_query_chars]

        request_id = metadata.request_id
        created = int(time.time())

        static_match = self.static_gate.match(normalized_query, persona_tag=metadata.persona_tag)
        if static_match is not None:
            return self._build_response(
                request=request,
                created=created,
                content=static_match.text,
                route=Route.STATIC_REPLY,
                reply_kind=ReplyKind.COMPLETE_STATIC_REPLY,
                continue_with_main_llm=False,
                rule_id=static_match.rule_id,
                model_version=None,
                fallback_reason=None,
                prompt_tokens=0,
                completion_tokens=0,
                request_id=request_id,
            )

        cache_key = (
            f"{normalized_query}:{metadata.persona_tag}:"
            f"{self.static_gate.version}:{NORMALIZER_VERSION}"
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._build_response(
                request=request,
                created=created,
                content=cached.text,
                route=Route(cached.route),
                reply_kind=ReplyKind(cached.reply_kind),
                continue_with_main_llm=cached.reply_kind == ReplyKind.FILLER_PREFIX.value,
                rule_id=cached.rule_id,
                model_version=cached.model_version,
                fallback_reason=FallbackReason(cached.fallback_reason) if cached.fallback_reason else None,
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
                request_id=request_id,
            )

        route = self.persona_router.resolve(metadata.persona_tag)
        if route is None:
            return self._fallback_response(
                request=request,
                created=created,
                persona_tag=metadata.persona_tag,
                locale=metadata.locale,
                fallback_reason=FallbackReason.MODEL_UNAVAILABLE,
                request_id=request_id,
            )

        generator = self.model_pool.get(route.bundle_dir)
        logger.info(
            "MiniMind3 input request_id=%s persona_tag=%s model_version=%s input=%r",
            request_id,
            metadata.persona_tag,
            route.model_version,
            normalized_query,
        )
        generation = generator.generate(normalized_query)
        logger.info(
            "MiniMind3 output request_id=%s persona_tag=%s model_version=%s output=%r "
            "timed_out=%s prompt_tokens=%s completion_tokens=%s",
            request_id,
            metadata.persona_tag,
            route.model_version,
            generation.text,
            generation.timed_out,
            generation.prompt_tokens,
            generation.completion_tokens,
        )
        if generation.timed_out:
            return self._fallback_response(
                request=request,
                created=created,
                persona_tag=metadata.persona_tag,
                locale=metadata.locale,
                fallback_reason=FallbackReason.TIMEOUT,
                request_id=request_id,
            )

        validation = generator.validator.validate(generation.text)
        if not validation.ok:
            logger.warning(
                "MiniMind3 output validation failed request_id=%s persona_tag=%s "
                "model_version=%s reason=%s query=%r output=%r",
                request_id,
                metadata.persona_tag,
                route.model_version,
                validation.reason.value if validation.reason else None,
                normalized_query,
                generation.text,
            )
            return self._fallback_response(
                request=request,
                created=created,
                persona_tag=metadata.persona_tag,
                locale=metadata.locale,
                fallback_reason=validation.reason or FallbackReason.UNSAFE_OUTPUT,
                request_id=request_id,
            )

        self.cache.set(
            cache_key,
            CachedFillerResult(
                text=generation.text,
                route=Route.MINIMIND3_FILLER.value,
                reply_kind=ReplyKind.FILLER_PREFIX.value,
                model_version=route.model_version,
                rule_id=None,
                fallback_reason=None,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
            ),
        )
        return self._build_response(
            request=request,
            created=created,
            content=generation.text,
            route=Route.MINIMIND3_FILLER,
            reply_kind=ReplyKind.FILLER_PREFIX,
            continue_with_main_llm=True,
            rule_id=None,
            model_version=route.model_version,
            fallback_reason=None,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            request_id=request_id,
        )

    def _fallback_response(
        self,
        *,
        request: ChatCompletionRequest,
        created: int,
        persona_tag: str,
        locale: str,
        fallback_reason: FallbackReason,
        request_id: str | None,
    ) -> ChatCompletionResponse:
        text = self.fallback_registry.choose(locale=locale, persona_tag=persona_tag)
        return self._build_response(
            request=request,
            created=created,
            content=text,
            route=Route.FALLBACK_FILLER,
            reply_kind=ReplyKind.FILLER_PREFIX,
            continue_with_main_llm=True,
            rule_id=None,
            model_version=None,
            fallback_reason=fallback_reason,
            prompt_tokens=0,
            completion_tokens=0,
            request_id=request_id,
        )

    def _build_response(
        self,
        *,
        request: ChatCompletionRequest,
        created: int,
        content: str,
        route: Route,
        reply_kind: ReplyKind,
        continue_with_main_llm: bool,
        rule_id: str | None,
        model_version: str | None,
        fallback_reason: FallbackReason | None,
        prompt_tokens: int,
        completion_tokens: int,
        request_id: str | None,
    ) -> ChatCompletionResponse:
        completion_id = f"chatcmpl-{request_id or created}"
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model or self.settings.service_model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=content),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            filler_reply=FillerReplyMeta(
                reply_kind=reply_kind,
                continue_with_main_llm=continue_with_main_llm,
                route=route,
                rule_id=rule_id,
                model_version=model_version,
                fallback_reason=fallback_reason,
                request_id=request_id,
            ),
        )
