from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from filler_words.core.enums import FallbackReason, ReplyKind, Route


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return value.strip()


class RequestMetadata(BaseModel):
    persona_tag: str = Field(..., min_length=1)
    locale: str = Field(default="zh-CN")
    request_id: str | None = None

    @field_validator("persona_tag", "locale")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    metadata: RequestMetadata | None = None

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model must not be blank")
        return stripped

    def extract_user_query(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user" and message.content:
                return message.content
        raise ValueError("messages must contain at least one non-empty user message")


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop"] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class FillerReplyMeta(BaseModel):
    reply_kind: ReplyKind
    continue_with_main_llm: bool
    route: Route
    rule_id: str | None = None
    model_version: str | None = None
    fallback_reason: FallbackReason | None = None
    request_id: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    filler_reply: FillerReplyMeta


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "flash-echo"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class OpenAIErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorBody


class ErrorDetail(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status_code: int
    message: str
    error_type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None

    def to_response(self) -> OpenAIErrorResponse:
        return OpenAIErrorResponse(
            error=OpenAIErrorBody(
                message=self.message,
                type=self.error_type,
                param=self.param,
                code=self.code,
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return self.to_response().model_dump()
