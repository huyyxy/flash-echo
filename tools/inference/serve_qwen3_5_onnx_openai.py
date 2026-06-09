"""OpenAI 兼容 HTTP 对话服务，后端使用 Qwen3.5-0.8B ONNX GenAI 部署包。

默认读取 ``tools/export/export_qwen3_5_onnx.py`` 导出的
``models/deploy/qwen3.5-0.8b-v1.0.0``。

依赖::

    pip install 'transformers>=5.3' fastapi 'uvicorn[standard]' pydantic
    pip install --pre onnxruntime-genai

启动示例::

    python3 tools/inference/serve_qwen3_5_onnx_openai.py

    python3 tools/inference/serve_qwen3_5_onnx_openai.py \\
      --model-dir models/deploy/qwen3.5-0.8b-v1.0.0 \\
      --host 0.0.0.0 \\
      --port 8000

调用示例::

    curl -s http://127.0.0.1:8000/v1/chat/completions \\
      -H 'content-type: application/json' \\
      -d '{
        "model": "qwen3.5-0.8b",
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}]
      }'

    curl -N http://127.0.0.1:8000/v1/chat/completions \\
      -H 'content-type: application/json' \\
      -d '{
        "model": "qwen3.5-0.8b",
        "messages": [{"role": "user", "content": "讲一个冷笑话"}],
        "stream": true
      }'
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Literal

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

_INFERENCE_DIR = Path(__file__).resolve().parent
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

import chat_qwen3_5_onnx as qwen_chat  # noqa: E402

PROJECT_ROOT = qwen_chat.PROJECT_ROOT
DEFAULT_MODEL_DIR = qwen_chat.DEFAULT_MODEL_DIR
DEFAULT_MODEL_NAME = "qwen3.5-0.8b"


@dataclass(frozen=True)
class ServerSettings:
    model_dir: Path
    model_name: str
    host: str
    port: int
    max_context_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    execution_provider: str
    enable_thinking: bool


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return value.strip()


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model must not be blank")
        return stripped

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("temperature must be >= 0")
        return value

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float | None) -> float | None:
        if value is not None and not (0 < value <= 1):
            raise ValueError("top_p must be in (0, 1]")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_tokens must be greater than 0")
        return value


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


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


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


@dataclass
class LoadedRuntime:
    og: Any
    model: Any
    genai_tokenizer: Any
    hf_tokenizer: Any
    model_version: str
    lock: threading.Lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Qwen3.5-0.8B ONNX chat over an OpenAI-compatible HTTP API."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"ONNX GenAI bundle directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model id exposed by /v1/models (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP listen host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP listen port.")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4096,
        help="Maximum total tokens kept for prompt plus generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Default max_tokens when the client omits it.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Default temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Default top_p.")
    parser.add_argument("--top-k", type=int, default=20, help="Default top_k.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty passed to onnxruntime-genai.",
    )
    parser.add_argument(
        "--execution-provider",
        default="cpu",
        help="ORT GenAI execution provider, for example cpu, cuda, dml, or webgpu.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode in chat template when the tokenizer supports it.",
    )
    return parser.parse_args()


def validate_server_args(args: argparse.Namespace) -> ServerSettings:
    qwen_chat.validate_genai_bundle_for_chat(args.model_dir)
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model_dir}")
    if not (args.model_dir / "genai_config.json").is_file():
        raise FileNotFoundError(
            f"missing genai_config.json in {args.model_dir}; "
            "please export with tools/export/export_qwen3_5_onnx.py first"
        )
    if args.max_context_tokens <= 0:
        raise ValueError("--max-context-tokens must be greater than 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than 0")
    if args.max_new_tokens >= args.max_context_tokens:
        raise ValueError("--max-new-tokens must be smaller than --max-context-tokens")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.top_p <= 0 or args.top_p > 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    model_name = args.model_name.strip()
    if not model_name:
        raise ValueError("--model-name must not be blank")

    return ServerSettings(
        model_dir=args.model_dir,
        model_name=model_name,
        host=args.host,
        port=args.port,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        execution_provider=args.execution_provider,
        enable_thinking=args.enable_thinking,
    )


def load_runtime(settings: ServerSettings) -> LoadedRuntime:
    og, AutoTokenizer = qwen_chat.import_runtime()
    model = qwen_chat.load_model(og, settings.model_dir, settings.execution_provider)
    genai_tokenizer = og.Tokenizer(model)
    hf_tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, trust_remote_code=True)
    model_version = qwen_chat.load_model_version(settings.model_dir)
    return LoadedRuntime(
        og=og,
        model=model,
        genai_tokenizer=genai_tokenizer,
        hf_tokenizer=hf_tokenizer,
        model_version=model_version,
        lock=threading.Lock(),
    )


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def build_inference_args(settings: ServerSettings, request: ChatCompletionRequest) -> SimpleNamespace:
    max_new_tokens = request.max_tokens if request.max_tokens is not None else settings.max_new_tokens
    if max_new_tokens >= settings.max_context_tokens:
        raise ValueError("max_tokens must be smaller than the server max context window")

    return SimpleNamespace(
        max_context_tokens=settings.max_context_tokens,
        max_new_tokens=max_new_tokens,
        temperature=request.temperature if request.temperature is not None else settings.temperature,
        top_p=request.top_p if request.top_p is not None else settings.top_p,
        top_k=settings.top_k,
        repetition_penalty=settings.repetition_penalty,
        enable_thinking=settings.enable_thinking,
        no_stream=True,
    )


def generate_completion(
    runtime: LoadedRuntime,
    *,
    messages: list[dict[str, str]],
    inference_args: SimpleNamespace,
) -> qwen_chat.GenerationStats:
    max_prompt_tokens = inference_args.max_context_tokens - inference_args.max_new_tokens
    trimmed = qwen_chat.trim_messages_for_context(
        runtime.hf_tokenizer,
        messages,
        max_prompt_tokens=max_prompt_tokens,
        enable_thinking=inference_args.enable_thinking,
    )
    prompt = qwen_chat.apply_chat_template(
        runtime.hf_tokenizer,
        trimmed,
        enable_thinking=inference_args.enable_thinking,
    )
    input_ids = list(runtime.genai_tokenizer.encode(prompt))
    params = runtime.og.GeneratorParams(runtime.model)
    params.set_search_options(
        **qwen_chat.build_search_options(inference_args, prompt_tokens=len(input_ids))
    )

    generator = runtime.og.Generator(runtime.model, params)
    qwen_chat.set_generator_inputs(generator, params, input_ids)
    tokenizer_stream = runtime.genai_tokenizer.create_stream()
    stop_ids = qwen_chat.stop_token_ids(runtime.hf_tokenizer)

    pieces: list[str] = []
    while not generator.is_done():
        generator.generate_next_token()
        new_tokens = generator.get_next_tokens()
        if not new_tokens:
            continue
        token_id = int(new_tokens[0])
        if token_id in stop_ids:
            break
        pieces.append(tokenizer_stream.decode(token_id))

    text, completion_tokens = qwen_chat.decode_generated_text(
        runtime.hf_tokenizer,
        generator,
        prompt_tokens=len(input_ids),
        streamed_text="".join(pieces),
    )
    if completion_tokens == 0:
        completion_tokens = len(pieces)

    return qwen_chat.GenerationStats(
        text=text,
        prompt_tokens=len(input_ids),
        completion_tokens=completion_tokens,
    )


def stream_completion_tokens(
    runtime: LoadedRuntime,
    *,
    messages: list[dict[str, str]],
    inference_args: SimpleNamespace,
) -> tuple[int, Iterator[str]]:
    max_prompt_tokens = inference_args.max_context_tokens - inference_args.max_new_tokens
    trimmed = qwen_chat.trim_messages_for_context(
        runtime.hf_tokenizer,
        messages,
        max_prompt_tokens=max_prompt_tokens,
        enable_thinking=inference_args.enable_thinking,
    )
    prompt = qwen_chat.apply_chat_template(
        runtime.hf_tokenizer,
        trimmed,
        enable_thinking=inference_args.enable_thinking,
    )
    input_ids = list(runtime.genai_tokenizer.encode(prompt))
    params = runtime.og.GeneratorParams(runtime.model)
    params.set_search_options(
        **qwen_chat.build_search_options(inference_args, prompt_tokens=len(input_ids))
    )

    generator = runtime.og.Generator(runtime.model, params)
    qwen_chat.set_generator_inputs(generator, params, input_ids)
    tokenizer_stream = runtime.genai_tokenizer.create_stream()
    stop_ids = qwen_chat.stop_token_ids(runtime.hf_tokenizer)

    def iter_tokens() -> Iterator[str]:
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if not new_tokens:
                continue
            token_id = int(new_tokens[0])
            if token_id in stop_ids:
                break
            yield tokenizer_stream.decode(token_id)

    return len(input_ids), iter_tokens()


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def error_payload(
    *,
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return OpenAIErrorResponse(
        error=OpenAIErrorBody(
            message=message,
            type=error_type,
            param=param,
            code=code,
        )
    ).model_dump()


def create_app(settings: ServerSettings, runtime: LoadedRuntime) -> FastAPI:
    app = FastAPI(title="Qwen3.5 ONNX OpenAI Server", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models() -> ModelListResponse:
        created = int(time.time())
        return ModelListResponse(
            data=[
                ModelCard(
                    id=settings.model_name,
                    created=created,
                )
            ]
        )

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest = Body(...)) -> Any:
        try:
            inference_args = build_inference_args(settings, request)
            message_dicts = messages_to_dicts(request.messages)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=error_payload(message=str(exc)),
            ) from exc

        completion_id = new_completion_id()
        created = int(time.time())

        if request.stream:
            return _stream_chat_completion(
                runtime=runtime,
                settings=settings,
                request=request,
                inference_args=inference_args,
                message_dicts=message_dicts,
                completion_id=completion_id,
                created=created,
            )

        with runtime.lock:
            stats = generate_completion(
                runtime,
                messages=message_dicts,
                inference_args=inference_args,
            )

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=settings.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=stats.text),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens,
                total_tokens=stats.prompt_tokens + stats.completion_tokens,
            ),
        )

    @app.exception_handler(HTTPException)
    async def openai_http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            payload = exc.detail
        else:
            payload = error_payload(message=str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=payload)

    return app


def _stream_chat_completion(
    *,
    runtime: LoadedRuntime,
    settings: ServerSettings,
    request: ChatCompletionRequest,
    inference_args: SimpleNamespace,
    message_dicts: list[dict[str, str]],
    completion_id: str,
    created: int,
) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        role_sent = False
        completion_tokens = 0

        with runtime.lock:
            prompt_tokens, token_iter = stream_completion_tokens(
                runtime,
                messages=message_dicts,
                inference_args=inference_args,
            )

            for piece in token_iter:
                completion_tokens += 1
                delta: dict[str, Any] = {}
                if not role_sent:
                    delta["role"] = "assistant"
                    role_sent = True
                if piece:
                    delta["content"] = piece

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": settings.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main() -> int:
    args = parse_args()
    settings = validate_server_args(args)
    runtime = load_runtime(settings)
    app = create_app(settings, runtime)

    print(
        f"loaded model={runtime.model_version} from {settings.model_dir}\n"
        f"listening on http://{settings.host}:{settings.port}\n"
        f"model id: {settings.model_name}"
    )
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
