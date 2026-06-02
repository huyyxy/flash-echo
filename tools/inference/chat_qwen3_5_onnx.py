"""使用 Qwen3.5-0.8B ONNX GenAI 部署包进行对话推理。

默认读取 ``tools/export/export_qwen3_5_onnx.py`` 导出的
``models/deploy/qwen3.5-0.8b-v1.0.0``。

依赖::

    pip install 'transformers>=5.3'
    pip install --pre onnxruntime-genai

使用示例::

    # 单轮对话
    python3 tools/inference/chat_qwen3_5_onnx.py --prompt "用一句话介绍你自己"

    # 交互式多轮对话
    python3 tools/inference/chat_qwen3_5_onnx.py

    # 指定部署包与采样参数
    python3 tools/inference/chat_qwen3_5_onnx.py \\
      --model-dir models/deploy/qwen3.5-0.8b-v1.0.0 \\
      --max-new-tokens 256 \\
      --temperature 0.7 \\
      --top-p 0.8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models/deploy/qwen3.5-0.8b-v1.0.0"


@dataclass(frozen=True)
class GenerationStats:
    text: str
    prompt_tokens: int
    completion_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a Qwen3.5-0.8B ONNX bundle exported by onnxruntime-genai."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"ONNX GenAI bundle directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run one request and exit. Without this argument, starts an interactive chat.",
    )
    parser.add_argument(
        "--system",
        default="你是一个有帮助的中文助手。",
        help="Optional system prompt. Use an empty string to disable it.",
    )
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
        help="Maximum tokens to generate for each assistant response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0 for greedy-like decoding when supported.",
    )
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling top_p.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling value.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty passed to onnxruntime-genai.",
    )
    parser.add_argument(
        "--execution-provider",
        default="cpu",
        help=(
            "ORT GenAI execution provider, for example cpu, cuda, dml, or webgpu. "
            "Use follow_config to keep providers from genai_config.json."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode in chat template when the tokenizer supports it.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming and print the full answer after generation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
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
    validate_genai_bundle_for_chat(args.model_dir)


def read_genai_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "genai_config.json").read_text(encoding="utf-8"))


def validate_genai_bundle_for_chat(model_dir: Path) -> None:
    cfg = read_genai_config(model_dir)
    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"invalid genai_config.json in {model_dir}: missing model section")

    decoder = model.get("decoder")
    if not isinstance(decoder, dict):
        raise ValueError(f"invalid genai_config.json in {model_dir}: missing decoder section")

    inputs = decoder.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"invalid genai_config.json in {model_dir}: missing decoder.inputs section")

    uses_inputs_embeds = "inputs_embeds" in inputs and "input_ids" not in inputs
    if uses_inputs_embeds and "embedding" not in model:
        raise RuntimeError(
            "This ONNX bundle was exported as decoder-only inputs_embeds "
            "(exclude_embeds=true) and cannot be loaded for CPU chat inference. "
            "Re-export with:\n"
            "  python3 tools/export/export_qwen3_5_onnx.py --overwrite\n"
            "The export script defaults to exclude_embeds=false for text chat."
        )


def import_runtime() -> tuple[Any, Any]:
    try:
        import onnxruntime_genai as og
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 ONNX chat requires onnxruntime-genai. "
            "Install with: pip install --pre onnxruntime-genai"
        ) from exc

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 chat template requires transformers>=5.3. "
            "Install with: pip install 'transformers>=5.3'"
        ) from exc

    return og, AutoTokenizer


def load_model(og: Any, model_dir: Path, execution_provider: str) -> Any:
    config = og.Config(str(model_dir))
    if execution_provider != "follow_config":
        config.clear_providers()
        if execution_provider != "cpu":
            config.append_provider(execution_provider)

    try:
        return og.Model(config)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load {model_dir} with execution provider {execution_provider!r}"
        ) from exc


def load_model_version(model_dir: Path) -> str:
    card_path = model_dir / "model_card.json"
    if not card_path.is_file():
        return model_dir.name
    try:
        payload = json.loads(card_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return model_dir.name
    if isinstance(payload, dict) and isinstance(payload.get("model_version"), str):
        return payload["model_version"]
    return model_dir.name


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            **kwargs,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded.input_ids)


def trim_messages_for_context(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_prompt_tokens: int,
    enable_thinking: bool,
) -> list[dict[str, str]]:
    trimmed = list(messages)
    while len(trimmed) > 1:
        prompt = apply_chat_template(
            tokenizer,
            trimmed,
            enable_thinking=enable_thinking,
        )
        if token_count(tokenizer, prompt) <= max_prompt_tokens:
            return trimmed

        drop_index = 1 if trimmed[0]["role"] == "system" and len(trimmed) > 2 else 0
        del trimmed[drop_index]

    return trimmed


def build_search_options(args: argparse.Namespace, *, prompt_tokens: int) -> dict[str, Any]:
    max_length = min(args.max_context_tokens, prompt_tokens + args.max_new_tokens)
    options: dict[str, Any] = {
        "max_length": max_length,
        "repetition_penalty": float(args.repetition_penalty),
    }
    if args.temperature > 0:
        options["temperature"] = float(args.temperature)
        options["top_p"] = float(args.top_p)
        if args.top_k > 0:
            options["top_k"] = int(args.top_k)
    return options


def set_generator_inputs(generator: Any, params: Any, input_ids: list[int]) -> None:
    if hasattr(generator, "append_tokens"):
        generator.append_tokens(input_ids)
        return
    params.input_ids = input_ids


def decode_generated_text(
    hf_tokenizer: Any,
    generator: Any,
    *,
    prompt_tokens: int,
    streamed_text: str,
) -> tuple[str, int]:
    if hasattr(generator, "get_sequence"):
        sequence = list(generator.get_sequence(0))
        completion_ids = sequence[prompt_tokens:]
        text = hf_tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return text, len(completion_ids)

    text = streamed_text.strip()
    return text, 0


def generate_once(
    *,
    og: Any,
    model: Any,
    genai_tokenizer: Any,
    hf_tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
) -> GenerationStats:
    input_ids = list(genai_tokenizer.encode(prompt))
    params = og.GeneratorParams(model)
    params.set_search_options(**build_search_options(args, prompt_tokens=len(input_ids)))

    generator = og.Generator(model, params)
    set_generator_inputs(generator, params, input_ids)
    tokenizer_stream = genai_tokenizer.create_stream()

    pieces: list[str] = []
    while not generator.is_done():
        generator.generate_next_token()
        new_tokens = generator.get_next_tokens()
        if not new_tokens:
            continue
        piece = tokenizer_stream.decode(new_tokens[0])
        pieces.append(piece)
        if not args.no_stream:
            print(piece, end="", flush=True)

    text, completion_tokens = decode_generated_text(
        hf_tokenizer,
        generator,
        prompt_tokens=len(input_ids),
        streamed_text="".join(pieces),
    )
    if args.no_stream:
        print(text)
    elif pieces:
        print()

    if completion_tokens == 0:
        completion_tokens = len(pieces)

    return GenerationStats(
        text=text,
        prompt_tokens=len(input_ids),
        completion_tokens=completion_tokens,
    )


def run_turn(
    *,
    og: Any,
    model: Any,
    genai_tokenizer: Any,
    hf_tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], GenerationStats]:
    max_prompt_tokens = args.max_context_tokens - args.max_new_tokens
    trimmed = trim_messages_for_context(
        hf_tokenizer,
        messages,
        max_prompt_tokens=max_prompt_tokens,
        enable_thinking=args.enable_thinking,
    )
    prompt = apply_chat_template(
        hf_tokenizer,
        trimmed,
        enable_thinking=args.enable_thinking,
    )
    stats = generate_once(
        og=og,
        model=model,
        genai_tokenizer=genai_tokenizer,
        hf_tokenizer=hf_tokenizer,
        prompt=prompt,
        args=args,
    )
    trimmed.append({"role": "assistant", "content": stats.text})
    return trimmed, stats


def initial_messages(system_prompt: str) -> list[dict[str, str]]:
    if system_prompt.strip():
        return [{"role": "system", "content": system_prompt.strip()}]
    return []


def run_single_prompt(args: argparse.Namespace, runtime: tuple[Any, Any, Any, Any]) -> None:
    og, model, genai_tokenizer, hf_tokenizer = runtime
    messages = initial_messages(args.system)
    messages.append({"role": "user", "content": args.prompt})
    _, stats = run_turn(
        og=og,
        model=model,
        genai_tokenizer=genai_tokenizer,
        hf_tokenizer=hf_tokenizer,
        messages=messages,
        args=args,
    )
    print(f"[usage] prompt_tokens={stats.prompt_tokens} completion_tokens={stats.completion_tokens}")


def run_interactive(args: argparse.Namespace, runtime: tuple[Any, Any, Any, Any]) -> None:
    og, model, genai_tokenizer, hf_tokenizer = runtime
    messages = initial_messages(args.system)

    print("进入 Qwen3.5 ONNX 对话模式。输入 /exit 退出，/clear 清空历史。")
    while True:
        try:
            user_text = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return
        if user_text == "/clear":
            messages = initial_messages(args.system)
            print("历史已清空。")
            continue

        messages.append({"role": "user", "content": user_text})
        print("Assistant> ", end="", flush=True)
        messages, stats = run_turn(
            og=og,
            model=model,
            genai_tokenizer=genai_tokenizer,
            hf_tokenizer=hf_tokenizer,
            messages=messages,
            args=args,
        )
        print(
            f"[usage] prompt_tokens={stats.prompt_tokens} "
            f"completion_tokens={stats.completion_tokens}"
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    og, AutoTokenizer = import_runtime()

    model = load_model(og, args.model_dir, args.execution_provider)
    genai_tokenizer = og.Tokenizer(model)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model_version = load_model_version(args.model_dir)
    print(f"loaded model={model_version} from {args.model_dir}")

    runtime = (og, model, genai_tokenizer, hf_tokenizer)
    if args.prompt is not None:
        run_single_prompt(args, runtime)
    else:
        run_interactive(args, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
