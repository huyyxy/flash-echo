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
    python3 tools/training/prepare_sharegpt_filler_prefix.py

    # 指定 Persona 与输出目录
    python3 tools/training/prepare_sharegpt_filler_prefix.py \\
      --persona female_receptionist \\
      --output-dir data/filler_prefix/female_receptionist

    # 小规模试跑
    python3 tools/training/prepare_sharegpt_filler_prefix.py --max-records 50 --overwrite

    # 断点续跑（默认）：已写入 output-dir 的 query 会跳过
    python3 tools/training/prepare_sharegpt_filler_prefix.py --output-dir data/filler_prefix/grandpa

    # 每成功写入 50 条打印一次进度（0 表示关闭）；skipped 仅在启动前 scan summary 中打印
    python3 tools/training/prepare_sharegpt_filler_prefix.py --log-interval 50

    # 并发调用 LLM（文件写入仍在主线程单线程完成）
    python3 tools/training/prepare_sharegpt_filler_prefix.py --workers 8
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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

VALID_PREFIX_SUFFIXES = ("，", "。", "！", "？", ",", ".", "!", "?", "……", ":", "：")
MIN_PREFIX_CHARS = 2
PREFERRED_MAX_PREFIX_CHARS = 18
MAX_PREFIX_CHARS = 25
ALLOWED_SINGLE_CHAR_INTERJECTIONS = frozenset({"嗯", "啊", "哦", "呃", "唉", "哈", "诶", "额"})

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
            "Print progress every N processed samples (written + generation failures). "
            f"Default {PROGRESS_LOG_INTERVAL}. Use 0 to disable. "
            "Skipped counters appear only in the input scan summary."
        ),
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=None,
        help=(
            "Append permanently failed samples to this JSONL file. "
            "Defaults to {output_dir}/failed.jsonl. Failed queries are skipped on resume."
        ),
    )
    parser.add_argument(
        "--no-failed-log",
        action="store_true",
        help="Do not write or resume from a failed-sample log.",
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


def prefix_stem_without_suffix(prefix: str) -> str:
    stem = prefix
    while stem and unicodedata.category(stem[-1]).startswith("P"):
        stem = stem[:-1]
    return stem


def is_allowed_short_interjection(prefix: str) -> bool:
    stem = prefix_stem_without_suffix(prefix)
    return len(stem) == 1 and stem in ALLOWED_SINGLE_CHAR_INTERJECTIONS


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
    if char_count < MIN_PREFIX_CHARS and not is_allowed_short_interjection(prefix):
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


def load_failed_queries(failed_log: Path) -> set[str]:
    failed: set[str] = set()
    if not failed_log.exists():
        return failed
    with failed_log.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip invalid failed log {failed_log}:{line_no}: {exc}", file=sys.stderr)
                continue
            query = record.get("query")
            if isinstance(query, str):
                failed.add(query_key(normalize_text(query)))
    return failed


def append_failed_records(
    failed_log: Path,
    failures: list[tuple[int, str, str]],
) -> None:
    if not failures:
        return
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    with failed_log.open("a", encoding="utf-8") as outfile:
        for line_no, query, error in failures:
            record = {"line_no": line_no, "query": query, "error": error}
            outfile.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        outfile.flush()


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

        "任务说明：\n"
        "你生成的内容不是正式回答，而是用户说完话后、主模型开始回答前播放的一句自然承接语。"
        "它的作用是让对话不中断，同时给主模型后续回答留下自然衔接空间。\n\n"

        "要求：\n"
        "1. 只输出一句短前缀，不要解释，不要输出 Markdown。\n"
        f"2. 长度优先控制在 {MIN_PREFIX_CHARS} 到 {PREFERRED_MAX_PREFIX_CHARS} 个实义字，"
        f"最长不超过 {MAX_PREFIX_CHARS} 个实义字。\n"
        "3. 优先生成自然、口语化、可被 TTS 立即播放的短句。\n"
        "4. 优先使用带衔接感的表达，例如："
        "「我先帮你梳理一下，」「这个问题可以这样看，」「我先抓重点说，」。\n"
        "5. 避免未说完的名词短语或残缺句，例如："
        "「昨天的电影是」「这个订单的状态是」「你的问题主要是」。\n"
        "6. 句末必须以逗号、句号、感叹号或问号结尾，"
        "中文 ，。！？ 或英文 , . ! ? 均可。\n"
        "7. 不要直接回答用户问题，不要给出方案、结论、原因、步骤或建议。\n"
        "8. 不要编造事实、数据、时间、地点、人物、状态或外部信息。\n"
        "9. 不要说未发生的操作，例如："
        "「我查到了」「我已经帮你处理好了」「我看了一下你的订单」「我已经确认了」。\n"
        "10. 不要做结果承诺，例如："
        "「我一定帮你解决」「这个肯定没问题」「马上就能处理好」。\n"
        "11. 不要输出会改变对话方向的反问句，例如："
        "「你是想问这个吗？」「你能再说清楚一点吗？」。\n"
        "12. 输出必须能让主模型从后面自然继续回答。\n"
        "13. 表达要符合 Persona，不要出现与 Persona 不一致的语气。\n"
        "14. 尽量避免所有 query 都生成同一种模板，保持自然多样性。\n"
        "15. 必须为每个输入 id 输出且只输出一条 filler_prefix，id 必须与输入一致。\n"
        "16. 只输出 JSON 对象，不要输出 Markdown、代码块或解释性文字。\n\n"

        "好例子：\n"
        "用户问技术方案时：我先按重点帮你拆，\n"
        "用户问复杂问题时：这个问题可以这样看，\n"
        "用户请求帮忙时：好的，我先帮你看一下，\n"
        "用户表达困惑时：我先帮你理清楚，\n"
        "用户提出选择问题时：我先帮你对比一下，\n\n"

        "坏例子：\n"
        "你可以用流式推理降低延迟。\n"
        "原因主要是模型首字生成慢。\n"
        "我已经帮你查到了。\n"
        "我保证马上解决。\n"
        "你是不是想问推理优化？\n"
        "昨天的电影是\n\n"

        "输出格式必须是：\n"
        '{"prefixes":[{"id":0,"filler_prefix":"这个问题可以这样看，"}]}\n\n'

        "待生成 queries：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_prefix_id(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"prefix id must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    raise ValueError(f"prefix id must be an integer, got {value!r}")


def extract_chat_completion_content(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        raise ValueError("API response must be a JSON object")
    if "error" in parsed:
        error = parsed["error"]
        if isinstance(error, dict):
            message = error.get("message", error)
        else:
            message = error
        raise ValueError(f"API error response: {message}")

    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response missing choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("API response missing message")

    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            raise ValueError("API response content is empty")
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "".join(parts)
        raise ValueError("API response content list has no text parts")
    raise ValueError(f"API response content must be a string, got {type(content).__name__}")


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
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"API response is not valid JSON: {exc}") from exc
    return extract_chat_completion_content(parsed)


def parse_json_object(text: str | None) -> dict[str, Any]:
    if text is None:
        raise ValueError("LLM response text is empty")
    if not isinstance(text, str):
        raise ValueError(f"LLM response text must be a string, got {type(text).__name__}")
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


def parse_llm_prefix_map(response_text: str) -> dict[int, str]:
    parsed = parse_json_object(response_text)
    prefixes = parsed.get("prefixes")
    if not isinstance(prefixes, list):
        raise ValueError("LLM response must contain a prefixes list")

    result: dict[int, str] = {}
    for item in prefixes:
        if not isinstance(item, dict):
            raise ValueError("each prefix item must be an object")
        prefix_id = parse_prefix_id(item.get("id"))
        if prefix_id in result:
            continue

        filler_prefix = item.get("filler_prefix")
        if not isinstance(filler_prefix, str):
            raise ValueError(f"filler_prefix for id={prefix_id} must be a string")

        result[prefix_id] = normalize_filler_prefix(filler_prefix)
    return result


def validate_llm_prefixes_per_item(
    response_text: str,
    batch: list[tuple[int, str]],
    *,
    validate_prefix: bool,
) -> tuple[dict[int, str], dict[int, str]]:
    parsed = parse_llm_prefix_map(response_text)
    valid: dict[int, str] = {}
    invalid: dict[int, str] = {}
    for index in range(len(batch)):
        filler_prefix = parsed.get(index)
        if filler_prefix is None:
            line_no, query = batch[index]
            invalid[index] = f"missing prefix id={index}; line_no={line_no} query={query!r}"
            continue
        if validate_prefix:
            try:
                validate_filler_prefix(filler_prefix)
            except ValueError as exc:
                line_no, query = batch[index]
                invalid[index] = (
                    f"{exc}; id={index} line_no={line_no} query={query!r} "
                    f"filler_prefix={filler_prefix!r}"
                )
                continue
        valid[index] = filler_prefix
    return valid, invalid


def format_retry_errors(errors: dict[int, str], *, limit: int = 2) -> str:
    parts = [errors[index] for index in sorted(errors)[:limit]]
    if len(errors) > limit:
        parts.append(f"... and {len(errors) - limit} more")
    return "; ".join(parts)


RETRYABLE_EXCEPTIONS = (
    HTTPError,
    URLError,
    TimeoutError,
    OSError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    AttributeError,
    ValueError,
)


def _split_indices(indices: list[int]) -> tuple[list[int], list[int]]:
    mid = len(indices) // 2
    return indices[:mid], indices[mid:]


def _retry_indices(
    batch: list[tuple[int, str]],
    indices: list[int],
    *,
    reason: str,
    persona: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    retries: int,
    retry_wait: float,
    validate_prefix: bool,
    results: dict[int, str],
    last_errors: dict[int, str],
) -> None:
    if not indices:
        return

    if len(indices) > 1:
        left, right = _split_indices(indices)
        print(
            f"split {len(indices)} sample(s) after {reason}",
            file=sys.stderr,
        )
        time.sleep(retry_wait)
        _generate_indices_with_retries(
            batch,
            left,
            persona=persona,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_wait=retry_wait,
            validate_prefix=validate_prefix,
            results=results,
            last_errors=last_errors,
        )
        _generate_indices_with_retries(
            batch,
            right,
            persona=persona,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_wait=retry_wait,
            validate_prefix=validate_prefix,
            results=results,
            last_errors=last_errors,
        )
        return

    index = indices[0]
    last_errors[index] = reason
    if retries <= 1:
        return
    print(
        f"retry 1 sample after error on attempt 1/{retries}: {reason}",
        file=sys.stderr,
    )
    time.sleep(retry_wait)
    _generate_indices_with_retries(
        batch,
        [index],
        persona=persona,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        retries=retries - 1,
        retry_wait=retry_wait,
        validate_prefix=validate_prefix,
        results=results,
        last_errors=last_errors,
    )


def _generate_indices_with_retries(
    batch: list[tuple[int, str]],
    indices: list[int],
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
    results: dict[int, str],
    last_errors: dict[int, str],
) -> None:
    """Generate prefixes for ``indices`` (positions in ``batch``).

    On API/JSON failures with multiple pending samples, split the chunk and retry
    smaller sub-batches instead of resubmitting the whole chunk (saves tokens and
    improves JSON reliability). Only single-sample API failures consume retry budget.
    """
    if not indices:
        return

    remaining = [index for index in indices if index not in results]
    if not remaining:
        return

    sub_batch = [batch[index] for index in remaining]
    try:
        response_text = call_openai_compatible_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=build_prefix_prompt(sub_batch, persona=persona),
            temperature=temperature,
            timeout=timeout,
        )
        valid, invalid = validate_llm_prefixes_per_item(
            response_text,
            sub_batch,
            validate_prefix=validate_prefix,
        )
    except RETRYABLE_EXCEPTIONS as exc:
        _retry_indices(
            batch,
            remaining,
            reason=str(exc),
            persona=persona,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_wait=retry_wait,
            validate_prefix=validate_prefix,
            results=results,
            last_errors=last_errors,
        )
        return

    for local_index, filler_prefix in valid.items():
        results[remaining[local_index]] = filler_prefix

    invalid_indices: list[int] = []
    for local_index, error in invalid.items():
        original_index = remaining[local_index]
        last_errors[original_index] = error
        invalid_indices.append(original_index)

    if not invalid_indices:
        return

    if len(invalid_indices) > 1:
        _retry_indices(
            batch,
            invalid_indices,
            reason=(
                "validation failure: "
                f"{format_retry_errors({idx: last_errors[idx] for idx in invalid_indices})}"
            ),
            persona=persona,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_wait=retry_wait,
            validate_prefix=validate_prefix,
            results=results,
            last_errors=last_errors,
        )
        return

    if retries <= 1:
        return

    _retry_indices(
        batch,
        invalid_indices,
        reason=last_errors[invalid_indices[0]],
        persona=persona,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_wait=retry_wait,
        validate_prefix=validate_prefix,
        results=results,
        last_errors=last_errors,
    )


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
) -> tuple[dict[int, str], list[tuple[int, str, str]]]:
    results: dict[int, str] = {}
    last_errors: dict[int, str] = {}
    _generate_indices_with_retries(
        batch,
        list(range(len(batch))),
        persona=persona,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_wait=retry_wait,
        validate_prefix=validate_prefix,
        results=results,
        last_errors=last_errors,
    )
    failures = [
        (batch[index][0], batch[index][1], last_errors.get(index, "unknown error"))
        for index in range(len(batch))
        if index not in results
    ]
    return results, failures


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
) -> tuple[dict[int, str], list[tuple[int, str, str]]]:
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
    """在主线程单线程写入一批生成结果（仅写入 prefixes 中已有的样本）。"""
    written = 0
    for index in sorted(prefixes):
        if index < 0 or index >= len(batch):
            continue
        _, query = batch[index]
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
    skipped_failed: int = 0,
) -> None:
    print(
        "input scan summary: "
        f"pending_samples={pending_samples}; "
        f"pending_batches={pending_batches}; "
        f"batch_size={batch_size}; "
        f"completed_on_disk={completed_on_disk}; "
        f"skipped_short={skipped_short}; "
        f"skipped_duplicate={skipped_duplicate}; "
        f"skipped_completed={skipped_completed}; "
        f"skipped_failed={skipped_failed}",
        flush=True,
    )


def print_progress(
    *,
    processed: int,
    pending_samples: int,
    written: int,
    skipped_generation: int,
    split_counts: Counter[str],
    prefix_length_counts: Counter[str],
) -> None:
    print(
        f"processed {processed}/{pending_samples} samples "
        f"(written={written}, skipped_generation={skipped_generation}); "
        f"split_counts={dict(sorted(split_counts.items()))}; "
        f"prefix_length_counts={dict(sorted(prefix_length_counts.items()))}",
        flush=True,
    )


def collect_pending_queries(
    *,
    input_path: Path,
    completed_queries: set[str],
    failed_queries: set[str],
    min_query_chars: int,
    max_records: int,
    no_dedupe: bool,
) -> tuple[list[tuple[int, str]], int, int, int, int]:
    seen_queries: set[str] = set()
    skipped_short = 0
    skipped_duplicate = 0
    skipped_completed = 0
    skipped_failed = 0
    pending_items: list[tuple[int, str]] = []
    accepted = 0

    for line_no, raw_query in iter_user_queries(input_path):
        query = normalize_text(raw_query)
        key = query_key(query)
        if len(query) < min_query_chars:
            skipped_short += 1
            continue
        if key in completed_queries:
            skipped_completed += 1
            continue
        if key in failed_queries:
            skipped_failed += 1
            continue
        if not no_dedupe:
            if query in seen_queries:
                skipped_duplicate += 1
                continue
            seen_queries.add(query)

        accepted += 1
        pending_items.append((line_no, query))
        if max_records and accepted >= max_records:
            break

    return pending_items, skipped_short, skipped_duplicate, skipped_completed, skipped_failed


def main() -> int:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    validate_runtime_args(args)
    validate_llm_config(args)

    if not args.input.exists():
        raise FileNotFoundError(f"input file does not exist: {args.input}")

    failed_log_path: Path | None = None
    if not args.no_failed_log:
        failed_log_path = args.failed_log or (args.output_dir / "failed.jsonl")

    split_counts: Counter[str] = Counter()
    prefix_length_counts: Counter[str] = Counter()
    skipped_generation = 0
    completed_queries = set() if args.overwrite else load_completed_queries(args.output_dir)
    failed_queries = set() if args.overwrite or args.no_failed_log else load_failed_queries(failed_log_path)
    written = 0

    if completed_queries:
        print(f"resume enabled: found {len(completed_queries)} completed queries in {args.output_dir}")
    if failed_queries and failed_log_path is not None:
        print(f"resume enabled: found {len(failed_queries)} failed queries in {failed_log_path}")

    (
        pending_items,
        skipped_short,
        skipped_duplicate,
        skipped_completed,
        skipped_failed,
    ) = collect_pending_queries(
        input_path=args.input,
        completed_queries=completed_queries,
        failed_queries=failed_queries,
        min_query_chars=args.min_query_chars,
        max_records=args.max_records,
        no_dedupe=args.no_dedupe,
    )
    pending_samples = len(pending_items)
    pending_batches = (pending_samples + args.batch_size - 1) // args.batch_size if pending_samples else 0
    print(
        f"persona={args.persona}; input={args.input}; output_dir={args.output_dir}; "
        f"model={args.model}; workers={args.workers}",
        flush=True,
    )
    if failed_log_path is not None:
        print(f"failed_log={failed_log_path}", flush=True)
    print_input_scan_summary(
        pending_samples=pending_samples,
        pending_batches=pending_batches,
        batch_size=args.batch_size,
        completed_on_disk=len(completed_queries),
        skipped_short=skipped_short,
        skipped_duplicate=skipped_duplicate,
        skipped_completed=skipped_completed,
        skipped_failed=skipped_failed,
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
    max_in_flight = max(args.workers * 2, args.workers)
    batch_iter = batch_items(iter(pending_items), args.batch_size)
    processed_for_log = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_batch: dict[Any, list[tuple[int, str]]] = {}

            def submit_next_batch() -> None:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    return
                future = executor.submit(generate_batch_prefixes, batch, **llm_kwargs)
                future_to_batch[future] = batch

            def maybe_log_progress() -> None:
                nonlocal processed_for_log
                if args.log_interval <= 0:
                    return
                processed = written + skipped_generation
                if processed // args.log_interval <= processed_for_log // args.log_interval:
                    return
                processed_for_log = processed
                print_progress(
                    processed=processed,
                    pending_samples=pending_samples,
                    written=written,
                    skipped_generation=skipped_generation,
                    split_counts=split_counts,
                    prefix_length_counts=prefix_length_counts,
                )

            for _ in range(min(max_in_flight, pending_batches)):
                submit_next_batch()

            while future_to_batch:
                done, _ = wait(future_to_batch, return_when=FIRST_COMPLETED)
                for future in done:
                    batch = future_to_batch.pop(future)
                    try:
                        prefixes, failures = future.result()
                    except Exception as exc:
                        failures = [
                            (line_no, query, f"batch worker error: {exc}")
                            for line_no, query in batch
                        ]
                        prefixes = {}
                        print(
                            f"batch worker error ({len(batch)} sample(s)): {exc}",
                            file=sys.stderr,
                        )

                    if failures:
                        skipped_generation += len(failures)
                        if failed_log_path is not None:
                            append_failed_records(failed_log_path, failures)
                        for line_no, query, error in failures:
                            print(
                                f"skip sample after retries: line_no={line_no} "
                                f"query={query!r} error={error}",
                                file=sys.stderr,
                            )
                        maybe_log_progress()

                    if prefixes:
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
                        maybe_log_progress()

                    submit_next_batch()
    finally:
        for outfile in outputs.values():
            outfile.close()

    print(f"wrote {written}/{pending_samples} samples to {args.output_dir}")
    if skipped_generation:
        print(f"skipped_generation={skipped_generation} samples after retries")
    print(f"split_counts={dict(sorted(split_counts.items()))}")
    print(f"prefix_length_counts={dict(sorted(prefix_length_counts.items()))}")
    print_input_scan_summary(
        pending_samples=pending_samples,
        pending_batches=pending_batches,
        batch_size=args.batch_size,
        completed_on_disk=len(completed_queries),
        skipped_short=skipped_short,
        skipped_duplicate=skipped_duplicate,
        skipped_completed=skipped_completed,
        skipped_failed=skipped_failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
