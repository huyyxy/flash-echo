"""将 ShareGPT 风格 JSONL 转为 filler 分类器训练数据。

从原始语料中读取 ``conversations`` 数组，抽取 ``role`` 为 ``user`` 的轮次作为
``query``，经 OpenAI 兼容 Chat Completions API 打标后，按稳定哈希划分并写入
``train.jsonl`` / ``valid.jsonl`` / ``test.jsonl``（格式见 ``data/README.md``）。

输入单条记录示例::

    {
      "conversations": [
        {"role": "user", "content": "帮我写一封请假邮件"},
        {"role": "assistant", "content": "..."}
      ]
    }

输出单条记录示例::

    {
      "query": "帮我写一封请假邮件",
      "trigger": 1,
      "filler_type": "ACKNOWLEDGE",
      "source": "sharegpt",
      "label_method": "llm_openai_compatible",
      "label_reason": "..."
    }

环境变量（可与命令行参数混用，也会自动读取项目根目录 ``.env``）::

    OPENAI_API_KEY    API 密钥（也可用 ``--api-key``）
    OPENAI_BASE_URL   API 根地址，默认 https://api.openai.com/v1
    OPENAI_MODEL      模型名（也可用 ``--model``）

配置优先级：命令行参数 > 当前进程环境变量 > 项目根目录 ``.env`` > 内置默认值。

使用示例::

    # 默认：读取 data/raw/sft_t2t_mini.jsonl，写入 data/processed/
    python scripts/prepare_sharegpt_jsonl.py

    # 也可在项目根目录 .env 中配置：
    # OPENAI_API_KEY=your-api-key
    # OPENAI_BASE_URL=https://api.openai.com/v1
    # OPENAI_MODEL=gpt-4o-mini

    # 指定输入、输出目录与模型
    python scripts/prepare_sharegpt_jsonl.py \\
      --input data/raw/my_sharegpt.jsonl \\
      --output-dir data/processed/v2 \\
      --model gpt-4o-mini \\
      --api-key "$OPENAI_API_KEY"

    # 使用国内或自建 OpenAI 兼容网关
    export OPENAI_BASE_URL="https://your-gateway/v1"
    export OPENAI_MODEL="qwen-plus"
    python scripts/prepare_sharegpt_jsonl.py --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL"

    # 小规模试跑：最多处理 50 条，覆盖已有输出
    python scripts/prepare_sharegpt_jsonl.py --max-records 50 --overwrite

    # 自定义划分比例与随机种子（哈希切分，可复现）
    python scripts/prepare_sharegpt_jsonl.py \\
      --train-ratio 0.85 --valid-ratio 0.1 --test-ratio 0.05 --seed project-a

    # 断点续跑（默认）：已写入 output-dir 的 query 会跳过
    python scripts/prepare_sharegpt_jsonl.py --output-dir data/processed

    # 精简输出、不去重短句
    python scripts/prepare_sharegpt_jsonl.py --no-reason --no-dedupe --min-query-chars 1
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
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FILLER_TYPES = {
    "NONE",
    "ACKNOWLEDGE",
    "THINKING",
    "FRAME",
    "EMPATHY",
    "RETRIEVAL",
    "CLARIFY_LEADIN",
}

LABEL_PRIORITY = (
    "EMPATHY",
    "CLARIFY_LEADIN",
    "RETRIEVAL",
    "ACKNOWLEDGE",
    "FRAME",
    "THINKING",
    "NONE",
)

WHITESPACE_RE = re.compile(r"\s+")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
PROGRESS_LOG_INTERVAL = 10

# 关闭支持混合思考模式的模型的 reasoning/thinking 输出，避免干扰 JSON 打标。
# - 火山方舟: thinking.type=disabled
# - 百炼 / Qwen: enable_thinking=false
# - vLLM Qwen3: chat_template_kwargs.enable_thinking=false
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
        description="Convert ShareGPT-style JSONL conversations into filler classifier JSONL splits."
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
        default=Path("data/processed"),
        help="Directory where train.jsonl, valid.jsonl, and test.jsonl are written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="42", help="Seed string used by the stable hash splitter.")
    parser.add_argument("--source", default="sharegpt", help="Value written to the source field.")
    parser.add_argument("--label-method", default="llm_openai_compatible")
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
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument(
        "--no-reason",
        action="store_true",
        help="Do not write label_reason to output records.",
    )
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
    return parser.parse_args(argv)


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


def trigger_for_label(filler_type: str) -> int:
    if filler_type not in FILLER_TYPES:
        raise ValueError(f"unknown filler_type: {filler_type}")
    return 0 if filler_type == "NONE" else 1


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
                query = record.get("query")
                if isinstance(query, str):
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


def build_label_prompt(queries: list[tuple[int, str]]) -> str:
    payload = [{"id": index, "query": query} for index, (_, query) in enumerate(queries)]
    trigger_label_priority = tuple(label for label in LABEL_PRIORITY if label != "NONE")
    return (
        "你是实时语音交互系统的数据标注员。你的任务是为当前用户 query 之后、主回答开始之前"
        "要播放的一句很短垫话，选择一个最合适的垫话功能类别。\n\n"
        "判定原则：\n"
        "- 垫话不是答案本身，而是用于承接、缓冲、铺垫或澄清的短前缀。\n"
        "- 当前数据集只保留需要垫话的样本；即使 query 很简单，也必须选择最接近的非 NONE 类别。\n\n"
        "标签定义：\n"
        "- ACKNOWLEDGE: 承接确认型。适用于明确的任务、请求或指令，主回答前适合先表示已收到。\n"
        "- THINKING: 思考缓冲型。适用于需要推理、比较、解释、归纳或计算，但不一定需要结构化展开的 query。\n"
        "- FRAME: 结构铺垫型。适用于开放观点、方案设计、长回答、步骤规划或需要先搭框架的 query。\n"
        "- EMPATHY: 情绪承接型。适用于用户表达情绪、压力、困惑、担忧、挫败或主观感受。\n"
        "- RETRIEVAL: 检索/回忆型。适用于明确要求查找资料、搜索信息、总结给定材料，或回忆已知上下文事实的 query。\n"
        "- CLARIFY_LEADIN: 澄清引导型。适用于信息不足、指代不明、目标不清，或存在多种合理解释、需要先问清楚的 query。\n\n"
        "约束：\n"
        "1. 只根据 query 文本判断，不假设额外上下文。\n"
        "2. trigger 必须为 1，filler_type 不能为 NONE。\n"
        "3. 不要输出 trigger=0 或 filler_type=NONE 的标签。\n"
        f"4. 多个标签都可能成立时，按优先级选择：{' > '.join(trigger_label_priority)}。\n"
        "5. 必须为每个输入 id 输出且只输出一条 label，id 必须与输入一致。\n"
        "6. reason 用一句简短中文说明标注依据，不要复述完整 query。\n"
        "7. 只输出 JSON 对象，不要输出 Markdown 或解释性文字。\n\n"
        "输出格式必须是：\n"
        '{"labels":[{"id":0,"trigger":1,"filler_type":"ACKNOWLEDGE","reason":"一句话说明标注依据"}]}\n\n'
        "待标注 queries：\n"
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


def label_batch_with_retries(
    *,
    batch: list[tuple[int, str]],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    retries: int,
    retry_wait: float,
) -> dict[int, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response_text = call_openai_compatible_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=build_label_prompt(batch),
                temperature=temperature,
                timeout=timeout,
            )
            return validate_llm_labels(response_text, batch)
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
    raise RuntimeError(f"failed to label batch after {retries} attempts: {last_error}")


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


def validate_llm_labels(response_text: str, batch: list[tuple[int, str]]) -> dict[int, dict[str, Any]]:
    parsed = parse_json_object(response_text)
    labels = parsed.get("labels")
    if not isinstance(labels, list):
        raise ValueError("LLM response must contain a labels list")

    expected_ids = set(range(len(batch)))
    result: dict[int, dict[str, Any]] = {}
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("each label item must be an object")
        label_id = item.get("id")
        if not isinstance(label_id, int) or isinstance(label_id, bool):
            raise ValueError("label id must be an integer")
        if label_id in result:
            raise ValueError(f"duplicate label id from LLM: {label_id}")
        filler_type = item.get("filler_type")
        trigger = item.get("trigger")
        if not isinstance(trigger, int) or isinstance(trigger, bool):
            raise ValueError(f"trigger for id={label_id} must be integer 0 or 1")
        if filler_type not in FILLER_TYPES:
            raise ValueError(f"unknown filler_type from LLM: {filler_type}")
        expected_trigger = trigger_for_label(filler_type)
        if trigger != expected_trigger:
            raise ValueError(
                f"inconsistent trigger for id={label_id}: "
                f"trigger={trigger}, filler_type={filler_type}"
            )
        result[label_id] = {
            "trigger": trigger,
            "filler_type": filler_type,
            "reason": str(item.get("reason", "")).strip(),
        }

    missing = expected_ids - result.keys()
    extra = result.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"LLM response ids mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return result


def validate_llm_config(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("missing API key. Set OPENAI_API_KEY or pass --api-key.")
    if not args.model:
        raise ValueError("missing model. Set OPENAI_MODEL or pass --model.")


def main() -> int:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    validate_runtime_args(args)
    validate_llm_config(args)

    if not args.input.exists():
        raise FileNotFoundError(f"input file does not exist: {args.input}")

    seen_queries: set[str] = set()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
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

    outputs = open_outputs(args.output_dir, append=not args.overwrite)
    try:
        for batch in batch_items(prepared_queries(), args.batch_size):
            labels = label_batch_with_retries(
                batch=batch,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
                retry_wait=args.retry_wait,
            )

            for index, (_, query) in enumerate(batch):
                label = labels[index]
                split = split_for_query(
                    query,
                    seed=args.seed,
                    train_ratio=args.train_ratio,
                    valid_ratio=args.valid_ratio,
                )
                record = {
                    "query": query,
                    "trigger": label["trigger"],
                    "filler_type": label["filler_type"],
                    "source": args.source,
                    "label_method": args.label_method,
                }
                if not args.no_reason:
                    record["label_reason"] = label["reason"]
                write_record(outputs[split], record)
                completed_queries.add(query_key(query))

                split_counts[split] += 1
                label_counts[label["filler_type"]] += 1
                written += 1
                if PROGRESS_LOG_INTERVAL > 0 and written % PROGRESS_LOG_INTERVAL == 0:
                    print(
                        f"processed {written} samples; "
                        f"split_counts={dict(sorted(split_counts.items()))}; "
                        f"label_counts={dict(sorted(label_counts.items()))}"
                    )
    finally:
        for outfile in outputs.values():
            outfile.close()

    print(f"wrote {written} samples to {args.output_dir}")
    print(f"split_counts={dict(sorted(split_counts.items()))}")
    print(f"label_counts={dict(sorted(label_counts.items()))}")
    print(
        f"skipped_short={skipped_short} "
        f"skipped_duplicate={skipped_duplicate} "
        f"skipped_completed={skipped_completed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
