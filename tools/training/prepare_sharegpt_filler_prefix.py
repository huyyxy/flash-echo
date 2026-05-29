"""将 ShareGPT 风格 JSONL 转为 MiniMind3 短垫话前缀（首句）训练数据。

从原始语料中读取 ``conversations`` 数组，抽取 ``role`` 为 ``user`` 的轮次作为
``query``，经 OpenAI 兼容 Chat Completions API 生成对应 Persona 风格的
``filler_prefix`` 后，按稳定哈希划分并写入 ``train.jsonl`` / ``valid.jsonl`` /
``test.jsonl``。输出为 MiniMind3 SFT 所需的 ``conversations`` 格式（与
``sft_t2t_mini.jsonl`` 一致），训练输入模板见
``docs/filler_words_model_implementation_steps.md`` 5.1。

输入单条记录示例::

    {
      "conversations": [
        {"role": "user", "content": "帮我写一封请假邮件"},
        {"role": "assistant", "content": "..."}
      ]
    }

输出单条记录示例::

    {
      "conversations": [
        {
          "role": "user",
          "content": "用户：帮我写一封请假邮件\n请生成一句可续写的短垫话前缀："
        },
        {"role": "assistant", "content": "好的，我来帮你整理一下，"}
      ]
    }

环境变量（可与命令行参数混用，也会自动读取项目根目录 ``.env``）::

    OPENAI_API_KEY    API 密钥（也可用 ``--api-key``）
    OPENAI_BASE_URL   API 根地址，默认 https://api.openai.com/v1
    OPENAI_MODEL      模型名（也可用 ``--model``）

配置优先级：命令行参数 > 当前进程环境变量 > 项目根目录 ``.env`` > 内置默认值。

使用示例::

    # 默认：读取 data/raw/sft_t2t_mini.jsonl，写入 data/filler_prefix/male_white_collar/
    python tools/training/prepare_sharegpt_filler_prefix.py

    # 指定 Persona 与输出目录
    python tools/training/prepare_sharegpt_filler_prefix.py \\
      --persona female_receptionist \\
      --output-dir data/filler_prefix/female_receptionist

    # 小规模试跑
    python tools/training/prepare_sharegpt_filler_prefix.py --max-records 50 --overwrite

    # 断点续跑（默认）：已写入 output-dir 的 query 会跳过
    python tools/training/prepare_sharegpt_filler_prefix.py --output-dir data/filler_prefix/grandpa

    # 每成功写入 50 条打印一次进度（0 表示关闭）；skipped 仅在启动前 scan summary 中打印
    python tools/training/prepare_sharegpt_filler_prefix.py --log-interval 50

    # 并发调用 LLM（文件写入仍在主线程单线程完成）
    python tools/training/prepare_sharegpt_filler_prefix.py --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PERSONA_DESCRIPTIONS: dict[str, str] = {
    "male_white_collar": "中年男白领，语气自然、干练、稳重，不夸张。",
    "female_receptionist": "前台女助理，语气自然、礼貌、温和，不夸张。",
    "grandpa": "老爷爷，语气亲切、平和、略带关怀，不夸张。",
    "young_girl": "女童，语气天真、活泼、简短，不夸张。",
}

VALID_PREFIX_SUFFIXES = ("，", "。", "！", "？", ",", ".", "!", "?")
MIN_PREFIX_CHARS = 3
PREFERRED_MAX_PREFIX_CHARS = 18
MAX_PREFIX_CHARS = 25

WHITESPACE_RE = re.compile(r"\s+")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_PERSONA = "male_white_collar"
PROGRESS_LOG_INTERVAL = 10
DEFAULT_WORKERS = 4
SFT_USER_PREFIX = "用户："
SFT_USER_PROMPT_SUFFIX = "请生成一句可续写的短垫话前缀："

LLM_REQUEST_DISABLE_THINKING: dict[str, Any] = {
    "thinking": {"type": "disabled"},
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def strip_dotenv_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double_quote:
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if char == "#" and not in_single_quote and not in_double_quote:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value.strip()


def parse_dotenv_value(raw_value: str) -> str:
    value = strip_dotenv_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if raw_value.strip().startswith('"'):
            value = (
                value.replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
    return value


def load_dotenv(dotenv_path: Path) -> dict[str, str]:
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    with dotenv_path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].lstrip()
            if "=" not in stripped:
                print(f"skip invalid .env line {line_no}: missing '='", file=sys.stderr)
                continue

            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            if not ENV_KEY_RE.fullmatch(key):
                print(f"skip invalid .env line {line_no}: invalid key {key!r}", file=sys.stderr)
                continue
            values[key] = parse_dotenv_value(raw_value)
    return values


def config_value(
    name: str,
    *,
    dotenv_values: dict[str, str],
    default: str | None = None,
) -> str | None:
    if name in os.environ:
        return os.environ[name]
    if name in dotenv_values:
        return dotenv_values[name]
    return default


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    dotenv_values = load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT-style JSONL conversations into filler-prefix SFT JSONL splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/sft_t2t_mini.jsonl"),
        help="Path to the raw ShareGPT-style JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where train/valid/test.jsonl are written. Defaults to data/filler_prefix/{persona}/.",
    )
    parser.add_argument(
        "--persona",
        choices=sorted(PERSONA_DESCRIPTIONS),
        default=DEFAULT_PERSONA,
        help="Persona tag whose style the generated filler_prefix should follow.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="42", help="Seed string used by the stable hash splitter.")
    parser.add_argument(
        "--api-key",
        default=config_value("OPENAI_API_KEY", dotenv_values=dotenv_values),
        help="OpenAI API key. Priority: CLI > environment > project .env.",
    )
    parser.add_argument(
        "--base-url",
        default=config_value(
            "OPENAI_BASE_URL",
            dotenv_values=dotenv_values,
            default=DEFAULT_OPENAI_BASE_URL,
        ),
        help="OpenAI-compatible API base URL. Priority: CLI > environment > project .env > default.",
    )
    parser.add_argument(
        "--model",
        default=config_value("OPENAI_MODEL", dotenv_values=dotenv_values),
        help="Model name. Priority: CLI > environment > project .env.",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of threads for concurrent LLM API calls. "
            "File writes always run on the main thread."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument(
        "--min-query-chars",
        type=int,
        default=2,
        help="Skip normalized user turns shorter than this many characters.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Stop after writing this many samples. 0 means no limit.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not exact-deduplicate normalized query text.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files instead of resuming from them.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip automatic filler_prefix validation before writing records.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=PROGRESS_LOG_INTERVAL,
        metavar="N",
        help=(
            "Print progress every N successfully written samples. "
            f"Default {PROGRESS_LOG_INTERVAL}. Use 0 to disable. "
            "Skipped counters appear only in the input scan summary."
        ),
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path("data/filler_prefix") / args.persona
    return args


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def is_user_turn(turn: dict[str, Any]) -> bool:
    role = turn.get("role")
    if isinstance(role, str):
        return role.lower() == "user"

    speaker = turn.get("from")
    if isinstance(speaker, str):
        return speaker.lower() in {"human", "user"}

    return False


def turn_content(turn: dict[str, Any]) -> str | None:
    for key in ("content", "value"):
        content = turn.get(key)
        if isinstance(content, str):
            return content
    return None


def normalize_filler_prefix(filler_prefix: str) -> str:
    """规范化垫话前缀：空白折叠与 Unicode 归一化，保留中英文句末标点。"""
    return normalize_text(filler_prefix)


def meaningful_char_count(text: str) -> int:
    count = 0
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        count += 1
    return count


def validate_filler_prefix(filler_prefix: str) -> None:
    prefix = normalize_filler_prefix(filler_prefix)
    if not prefix:
        raise ValueError("filler_prefix must not be empty")
    if any(char in prefix for char in "\n\r\t"):
        raise ValueError("filler_prefix must be a single line")
    if not prefix.endswith(VALID_PREFIX_SUFFIXES):
        raise ValueError(
            f"filler_prefix must end with one of {VALID_PREFIX_SUFFIXES!r}, got {prefix!r}"
        )
    char_count = meaningful_char_count(prefix)
    if char_count < MIN_PREFIX_CHARS:
        raise ValueError(f"filler_prefix too short: {char_count} meaningful chars")
    if char_count > MAX_PREFIX_CHARS:
        raise ValueError(f"filler_prefix too long: {char_count} meaningful chars")


def split_for_query(
    query: str,
    *,
    seed: str,
    train_ratio: float,
    valid_ratio: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{query}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + valid_ratio:
        return "valid"
    return "test"


def query_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def build_training_user_content(query: str) -> str:
    return f"{SFT_USER_PREFIX}{query}\n{SFT_USER_PROMPT_SUFFIX}"


def build_sft_record(query: str, filler_prefix: str) -> dict[str, Any]:
    return {
        "conversations": [
            {"role": "user", "content": build_training_user_content(query)},
            {"role": "assistant", "content": filler_prefix},
        ]
    }


def query_from_output_record(record: dict[str, Any]) -> str | None:
    query = record.get("query")
    if isinstance(query, str):
        return normalize_text(query)

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None

    first_turn = conversations[0]
    if not isinstance(first_turn, dict):
        return None
    if first_turn.get("role") != "user":
        return None

    content = first_turn.get("content")
    if not isinstance(content, str):
        return None
    if not content.startswith(SFT_USER_PREFIX):
        return None
    if not content.endswith(SFT_USER_PROMPT_SUFFIX):
        return None

    query = content[len(SFT_USER_PREFIX) : -len(SFT_USER_PROMPT_SUFFIX)].removesuffix("\n")
    return normalize_text(query)


def iter_user_queries(input_path: Path) -> Iterator[tuple[int, str]]:
    with input_path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip invalid json at line {line_no}: {exc}", file=sys.stderr)
                continue

            conversations = record.get("conversations")
            if not isinstance(conversations, list):
                print(f"skip line {line_no}: missing conversations list", file=sys.stderr)
                continue

            for turn in conversations:
                if not isinstance(turn, dict) or not is_user_turn(turn):
                    continue
                content = turn_content(turn)
                if content is not None:
                    yield line_no, content


def batch_items(items: Iterator[tuple[int, str]], batch_size: int) -> Iterator[list[tuple[int, str]]]:
    if batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")

    batch: list[tuple[int, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, valid_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")
    if args.workers <= 0:
        raise ValueError("workers must be greater than 0")
    if args.retries <= 0:
        raise ValueError("retries must be greater than 0")
    if args.retry_wait < 0:
        raise ValueError("retry-wait must be non-negative")
    if args.timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    if args.min_query_chars < 0:
        raise ValueError("min-query-chars must be non-negative")
    if args.max_records < 0:
        raise ValueError("max-records must be non-negative")
    if args.log_interval < 0:
        raise ValueError("log-interval must be non-negative")


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "train": output_dir / "train.jsonl",
        "valid": output_dir / "valid.jsonl",
        "test": output_dir / "test.jsonl",
    }


def load_completed_queries(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    for path in output_paths(output_dir).values():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as infile:
            for line_no, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"skip invalid existing output {path}:{line_no}: {exc}", file=sys.stderr)
                    continue
                query = query_from_output_record(record)
                if query is not None:
                    completed.add(query_key(query))
    return completed


def open_outputs(output_dir: Path, *, append: bool) -> dict[str, TextIO]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    return {
        split: path.open(mode, encoding="utf-8") for split, path in output_paths(output_dir).items()
    }


def write_record(outfile: TextIO, record: dict[str, Any]) -> None:
    outfile.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    outfile.flush()


def build_prefix_prompt(
    queries: list[tuple[int, str]],
    *,
    persona: str,
) -> str:
    persona_description = PERSONA_DESCRIPTIONS[persona]
    payload = [{"id": index, "query": query} for index, (_, query) in enumerate(queries)]
    return (
        "你是实时语音交互系统的数据合成助手。请根据用户 Query 生成一句中文短垫话前缀，"
        "用于主模型回答前立即播放。\n\n"
        f"Persona:\n{persona}，{persona_description}\n\n"
        "要求：\n"
        "1. 只输出一句短前缀，不要解释，不要输出 Markdown。\n"
        f"2. 长度优先控制在 {MIN_PREFIX_CHARS} 到 {PREFERRED_MAX_PREFIX_CHARS} 个中文字符，"
        f"最长不超过 {MAX_PREFIX_CHARS} 个中文字符。\n"
        "3. 句末必须以逗号、句号、感叹号或问号结尾（中文 ，。！？ 或英文 , . ! ? 均可）。\n"
        "4. 不要直接回答用户问题。\n"
        "5. 不要编造事实、数据、时间、地点或人物。\n"
        "6. 不要说“我查到了”“我已经帮你处理好了”等未发生的操作。\n"
        "7. 输出必须能让主模型从后面自然继续回答。\n"
        "8. 必须为每个输入 id 输出且只输出一条 filler_prefix，id 必须与输入一致。\n"
        "9. 只输出 JSON 对象，不要输出 Markdown 或解释性文字。\n\n"
        "输出格式必须是：\n"
        '{"prefixes":[{"id":0,"filler_prefix":"这个问题可以这样看，"}]}\n\n'
        "待生成 queries：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你只返回可解析的 JSON 对象，不返回 Markdown。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        **LLM_REQUEST_DISABLE_THINKING,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
    parsed = json.loads(response_body)
    return parsed["choices"][0]["message"]["content"]


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def validate_llm_prefixes(
    response_text: str,
    batch: list[tuple[int, str]],
    *,
    validate_prefix: bool,
) -> dict[int, str]:
    parsed = parse_json_object(response_text)
    prefixes = parsed.get("prefixes")
    if not isinstance(prefixes, list):
        raise ValueError("LLM response must contain a prefixes list")

    expected_ids = set(range(len(batch)))
    result: dict[int, str] = {}
    for item in prefixes:
        if not isinstance(item, dict):
            raise ValueError("each prefix item must be an object")
        prefix_id = item.get("id")
        if not isinstance(prefix_id, int) or isinstance(prefix_id, bool):
            raise ValueError("prefix id must be an integer")
        if prefix_id in result:
            raise ValueError(f"duplicate prefix id from LLM: {prefix_id}")

        filler_prefix = item.get("filler_prefix")
        if not isinstance(filler_prefix, str):
            raise ValueError(f"filler_prefix for id={prefix_id} must be a string")

        filler_prefix = normalize_filler_prefix(filler_prefix)
        if validate_prefix:
            validate_filler_prefix(filler_prefix)

        result[prefix_id] = filler_prefix

    missing = expected_ids - result.keys()
    extra = result.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"LLM response ids mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return result


def generate_batch_with_retries(
    *,
    batch: list[tuple[int, str]],
    persona: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    retries: int,
    retry_wait: float,
    validate_prefix: bool,
) -> dict[int, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response_text = call_openai_compatible_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=build_prefix_prompt(batch, persona=persona),
                temperature=temperature,
                timeout=timeout,
            )
            return validate_llm_prefixes(
                response_text,
                batch,
                validate_prefix=validate_prefix,
            )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                break
            print(f"retry batch after error on attempt {attempt}/{retries}: {exc}", file=sys.stderr)
            time.sleep(retry_wait * attempt)
    raise RuntimeError(f"failed to generate filler prefixes after {retries} attempts: {last_error}")


def generate_batch_prefixes(
    batch: list[tuple[int, str]],
    *,
    persona: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    retries: int,
    retry_wait: float,
    validate_prefix: bool,
) -> dict[int, str]:
    """仅调用 LLM，不写文件；供线程池并发执行。"""
    return generate_batch_with_retries(
        batch=batch,
        persona=persona,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_wait=retry_wait,
        validate_prefix=validate_prefix,
    )


def write_batch_records(
    batch: list[tuple[int, str]],
    prefixes: dict[int, str],
    *,
    outputs: dict[str, TextIO],
    completed_queries: set[str],
    seed: str,
    train_ratio: float,
    valid_ratio: float,
    split_counts: Counter[str],
    prefix_length_counts: Counter[str],
) -> int:
    """在主线程单线程写入一批生成结果。"""
    written = 0
    for index, (_, query) in enumerate(batch):
        filler_prefix = prefixes[index]
        split = split_for_query(
            query,
            seed=seed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
        record = build_sft_record(query, filler_prefix)
        write_record(outputs[split], record)
        completed_queries.add(query_key(query))

        split_counts[split] += 1
        char_count = meaningful_char_count(filler_prefix)
        if char_count <= PREFERRED_MAX_PREFIX_CHARS:
            prefix_length_counts["within_preferred"] += 1
        else:
            prefix_length_counts["over_preferred"] += 1
        written += 1
    return written


def validate_llm_config(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("missing API key. Set OPENAI_API_KEY or pass --api-key.")
    if not args.model:
        raise ValueError("missing model. Set OPENAI_MODEL or pass --model.")


def print_input_scan_summary(
    *,
    pending_samples: int,
    pending_batches: int,
    batch_size: int,
    completed_on_disk: int,
    skipped_short: int,
    skipped_duplicate: int,
    skipped_completed: int,
) -> None:
    print(
        "input scan summary: "
        f"pending_samples={pending_samples}; "
        f"pending_batches={pending_batches}; "
        f"batch_size={batch_size}; "
        f"completed_on_disk={completed_on_disk}; "
        f"skipped_short={skipped_short}; "
        f"skipped_duplicate={skipped_duplicate}; "
        f"skipped_completed={skipped_completed}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    validate_runtime_args(args)
    validate_llm_config(args)

    if not args.input.exists():
        raise FileNotFoundError(f"input file does not exist: {args.input}")

    seen_queries: set[str] = set()
    split_counts: Counter[str] = Counter()
    prefix_length_counts: Counter[str] = Counter()
    skipped_short = 0
    skipped_duplicate = 0
    skipped_completed = 0
    completed_queries = set() if args.overwrite else load_completed_queries(args.output_dir)
    written = 0

    def prepared_queries() -> Iterator[tuple[int, str]]:
        nonlocal skipped_short, skipped_duplicate, skipped_completed

        accepted = 0
        for line_no, raw_query in iter_user_queries(args.input):
            query = normalize_text(raw_query)
            key = query_key(query)
            if len(query) < args.min_query_chars:
                skipped_short += 1
                continue
            if key in completed_queries:
                skipped_completed += 1
                continue
            if not args.no_dedupe:
                if query in seen_queries:
                    skipped_duplicate += 1
                    continue
                seen_queries.add(query)

            accepted += 1
            yield line_no, query

            if args.max_records and accepted >= args.max_records:
                break

    if completed_queries:
        print(f"resume enabled: found {len(completed_queries)} completed queries in {args.output_dir}")

    batches = list(batch_items(prepared_queries(), args.batch_size))
    pending_samples = sum(len(batch) for batch in batches)
    print(
        f"persona={args.persona}; input={args.input}; output_dir={args.output_dir}; "
        f"model={args.model}; workers={args.workers}",
        flush=True,
    )
    print_input_scan_summary(
        pending_samples=pending_samples,
        pending_batches=len(batches),
        batch_size=args.batch_size,
        completed_on_disk=len(completed_queries),
        skipped_short=skipped_short,
        skipped_duplicate=skipped_duplicate,
        skipped_completed=skipped_completed,
    )

    outputs = open_outputs(args.output_dir, append=not args.overwrite)
    llm_kwargs = {
        "persona": args.persona,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "retries": args.retries,
        "retry_wait": args.retry_wait,
        "validate_prefix": not args.no_validate,
    }
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_batch = {
                executor.submit(generate_batch_prefixes, batch, **llm_kwargs): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                prefixes = future.result()
                prev_written = written
                written += write_batch_records(
                    batch,
                    prefixes,
                    outputs=outputs,
                    completed_queries=completed_queries,
                    seed=args.seed,
                    train_ratio=args.train_ratio,
                    valid_ratio=args.valid_ratio,
                    split_counts=split_counts,
                    prefix_length_counts=prefix_length_counts,
                )
                if (
                    args.log_interval > 0
                    and written // args.log_interval > prev_written // args.log_interval
                ):
                    print(
                        f"processed {written}/{pending_samples} samples; "
                        f"split_counts={dict(sorted(split_counts.items()))}; "
                        f"prefix_length_counts={dict(sorted(prefix_length_counts.items()))}",
                        flush=True,
                    )
    finally:
        for outfile in outputs.values():
            outfile.close()

    print(f"wrote {written}/{pending_samples} samples to {args.output_dir}")
    print(f"split_counts={dict(sorted(split_counts.items()))}")
    print(f"prefix_length_counts={dict(sorted(prefix_length_counts.items()))}")
    print_input_scan_summary(
        pending_samples=pending_samples,
        pending_batches=len(batches),
        batch_size=args.batch_size,
        completed_on_disk=len(completed_queries),
        skipped_short=skipped_short,
        skipped_duplicate=skipped_duplicate,
        skipped_completed=skipped_completed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
