"""审查模板文本能否自然作为每个 query 的回答开头。

脚本读取一个或多个 processed JSONL 文件，按每条记录的 ``filler_type`` 展开匹配的
模板，调用 OpenAI 兼容 Chat 模型判断每组 query/template pair，并只把不合适的 pair
写入 JSONL。

配置可以来自命令行参数、环境变量或项目根目录 ``.env`` 文件：

    OPENAI_API_KEY
    OPENAI_BASE_URL
    OPENAI_MODEL

示例：

    python3 scripts/review_template_starts.py \
      --input data/processed/train.jsonl \
      --output reports/template_start_issues.jsonl

    python3 scripts/review_template_starts.py \
      --input data/processed/*.jsonl \
      --all-personas \
      --max-records 20 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TEMPLATE_PATH = Path("configs/templates.zh-CN.v1.json")
DEFAULT_INPUT_PATH = Path("data/processed/demo.jsonl")
DEFAULT_OUTPUT_PATH = Path("reports/template_start_issues.jsonl")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WHITESPACE_RE = re.compile(r"\s+")
PROGRESS_LOG_INTERVAL = 100


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
                print(f"跳过无效 .env 行 {line_no}: 缺少 '='", file=sys.stderr)
                continue

            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            if not ENV_KEY_RE.fullmatch(key):
                print(f"跳过无效 .env 行 {line_no}: key 不合法 {key!r}", file=sys.stderr)
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
        description="审查已配置的垫话模板是否能作为 processed query 的回答开头。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT_PATH],
        help="待审查的 processed JSONL 文件，例如 data/processed/train.jsonl。",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="模板配置文件路径。",
    )
    parser.add_argument("--locale", default="zh-CN", help="模板配置中的 locale key。")
    parser.add_argument(
        "--persona",
        default="default",
        help="要审查的人设。传入 'default' 表示使用配置中的 default_persona。",
    )
    parser.add_argument(
        "--all-personas",
        action="store_true",
        help="审查所有人设下的模板，而不是只审查 --persona。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="不合适 query/template pair 的 JSONL 输出路径。",
    )
    parser.add_argument(
        "--api-key",
        default=config_value("OPENAI_API_KEY", dotenv_values=dotenv_values),
        help="OpenAI API key。优先级：命令行 > 环境变量 > 项目 .env。",
    )
    parser.add_argument(
        "--base-url",
        default=config_value(
            "OPENAI_BASE_URL",
            dotenv_values=dotenv_values,
            default=DEFAULT_OPENAI_BASE_URL,
        ),
        help="OpenAI 兼容 API 的 base URL。",
    )
    parser.add_argument(
        "--model",
        default=config_value("OPENAI_MODEL", dotenv_values=dotenv_values),
        help="模型名。优先级：命令行 > 环境变量 > 项目 .env。",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="最多审查多少条输入记录。0 表示不限制。",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="最多审查多少组 query/template pair。0 表示不限制。",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="同时审查 enabled=false 的模板。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果 --output 已存在则覆盖。",
    )
    return parser.parse_args(argv)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_args(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("缺少 API key。请设置 OPENAI_API_KEY 或传入 --api-key。")
    if not args.model:
        raise ValueError("缺少模型名。请设置 OPENAI_MODEL 或传入 --model。")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    if args.timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if args.retries <= 0:
        raise ValueError("retries 必须大于 0")
    if args.retry_wait < 0:
        raise ValueError("retry-wait 不能为负数")
    if args.max_records < 0:
        raise ValueError("max-records 不能为负数")
    if args.max_pairs < 0:
        raise ValueError("max-pairs 不能为负数")


def iter_processed_records(input_paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for input_path in input_paths:
        resolved_path = resolve_path(input_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"输入文件不存在：{resolved_path}")

        with resolved_path.open("r", encoding="utf-8") as infile:
            for line_no, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"跳过无效 JSON {resolved_path}:{line_no}: {exc}", file=sys.stderr)
                    continue
                if not isinstance(record, dict):
                    print(f"跳过非对象记录 {resolved_path}:{line_no}", file=sys.stderr)
                    continue

                query = record.get("query")
                filler_type = record.get("filler_type")
                if not isinstance(query, str) or not normalize_text(query):
                    print(f"跳过缺少 query 的记录 {resolved_path}:{line_no}", file=sys.stderr)
                    continue
                if not isinstance(filler_type, str) or not filler_type:
                    print(f"跳过缺少 filler_type 的记录 {resolved_path}:{line_no}", file=sys.stderr)
                    continue

                yield {
                    "source_path": str(input_path),
                    "line_no": line_no,
                    "query": normalize_text(query),
                    "filler_type": filler_type,
                    "record": record,
                }


def load_template_groups(
    template_path: Path,
    *,
    locale: str,
    persona: str,
    all_personas: bool,
    include_disabled: bool,
) -> dict[str, list[dict[str, Any]]]:
    resolved_path = resolve_path(template_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"模板配置文件不存在：{resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as infile:
        config = json.load(infile)

    locales = config.get("locales")
    if not isinstance(locales, dict) or locale not in locales:
        raise ValueError(f"模板配置中找不到 locale {locale!r}")

    locale_config = locales[locale]
    if not isinstance(locale_config, dict):
        raise ValueError(f"locale {locale!r} 必须对应一个人设对象")

    if all_personas:
        persona_names = list(locale_config.keys())
    else:
        persona_name = config.get("default_persona") if persona == "default" else persona
        if not isinstance(persona_name, str) or not persona_name:
            raise ValueError("模板配置缺少 default_persona")
        if persona_name not in locale_config:
            raise ValueError(f"locale {locale!r} 下找不到 persona {persona_name!r}")
        persona_names = [persona_name]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for persona_name in persona_names:
        persona_config = locale_config.get(persona_name)
        if not isinstance(persona_config, dict):
            continue
        for filler_type, templates in persona_config.items():
            if not isinstance(templates, list):
                continue
            for template in templates:
                if not isinstance(template, dict):
                    continue
                text = template.get("text")
                template_id = template.get("template_id")
                enabled = template.get("enabled", True)
                if not isinstance(text, str) or not text:
                    continue
                if not include_disabled and enabled is False:
                    continue
                grouped.setdefault(filler_type, []).append(
                    {
                        "persona": persona_name,
                        "template_id": template_id if isinstance(template_id, str) else "",
                        "text": normalize_text(text),
                        "enabled": enabled,
                    }
                )
    return grouped


def iter_review_pairs(
    records: Iterator[dict[str, Any]],
    template_groups: dict[str, list[dict[str, Any]]],
    *,
    max_records: int,
    max_pairs: int,
) -> Iterator[dict[str, Any]]:
    reviewed_records = 0
    yielded_pairs = 0
    for record in records:
        if max_records and reviewed_records >= max_records:
            break
        reviewed_records += 1

        templates = template_groups.get(record["filler_type"])
        if not templates:
            print(
                f"跳过 {record['source_path']}:{record['line_no']}: "
                f"没有 filler_type={record['filler_type']} 对应的模板",
                file=sys.stderr,
            )
            continue

        for template in templates:
            if max_pairs and yielded_pairs >= max_pairs:
                return
            yielded_pairs += 1
            yield {
                "source_path": record["source_path"],
                "line_no": record["line_no"],
                "query": record["query"],
                "filler_type": record["filler_type"],
                "persona": template["persona"],
                "template_id": template["template_id"],
                "text": template["text"],
                "template_enabled": template["enabled"],
            }


def batch_items(items: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_review_prompt(batch: list[dict[str, Any]]) -> str:
    payload = [
        {
            "id": index,
            "query": item["query"],
            "filler_type": item["filler_type"],
            "persona": item["persona"],
            "template_id": item["template_id"],
            "template_text": item["text"],
        }
        for index, item in enumerate(batch)
    ]
    return (
        "你是中文实时语音助手的数据质检员。请判断 template_text 是否可以作为 assistant 对"
        " query 的回答开头，也就是在正式回答内容之前先说出的短前缀。\n\n"
        "判定标准：\n"
        "- 如果 template_text 后面接一个正常回答时，语义、语气和任务承接都自然，则 suitable=true。\n"
        "- 它可以是简短垫话，不需要直接回答 query，但不能和用户意图冲突。\n"
        "- 不要因为 template_text 比较泛化就判错；只要能自然承接该 query 即可。\n"
        "- 如果它错误承诺了动作、检索、记忆、情绪理解、澄清需求，或会让回答开头显得突兀，"
        "则 suitable=false。\n"
        "- 特别注意 filler_type 是否与 query 场景匹配，例如情绪承接不能随便用于普通事实问题，"
        "检索/回忆引导不能用于纯写作执行类请求。\n\n"
        "只输出 JSON 对象，不要输出 Markdown。输出格式必须是：\n"
        '{"reviews":[{"id":0,"suitable":true,"reason":"一句简短中文原因"}]}\n\n'
        "待审查项目：\n"
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
        raise ValueError("LLM 响应必须是 JSON 对象")
    return parsed


def validate_reviews(response_text: str, batch: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    parsed = parse_json_object(response_text)
    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("LLM 响应必须包含 reviews 列表")

    expected_ids = set(range(len(batch)))
    result: dict[int, dict[str, Any]] = {}
    for item in reviews:
        if not isinstance(item, dict):
            raise ValueError("每个 review 条目都必须是对象")
        item_id = item.get("id")
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError("review id 必须是整数")
        if item_id in result:
            raise ValueError(f"LLM 返回了重复的 review id: {item_id}")

        suitable = item.get("suitable")
        if not isinstance(suitable, bool):
            raise ValueError(f"id={item_id} 的 suitable 必须是布尔值")
        result[item_id] = {
            "suitable": suitable,
            "reason": str(item.get("reason", "")).strip(),
        }

    missing = expected_ids - result.keys()
    extra = result.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"LLM 响应 id 不匹配：missing={sorted(missing)} extra={sorted(extra)}")
    return result


def review_batch_with_retries(
    *,
    batch: list[dict[str, Any]],
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
                prompt=build_review_prompt(batch),
                temperature=temperature,
                timeout=timeout,
            )
            return validate_reviews(response_text, batch)
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
            print(f"第 {attempt}/{retries} 次尝试出错，准备重试 batch: {exc}", file=sys.stderr)
            time.sleep(retry_wait * attempt)
    raise RuntimeError(f"重试 {retries} 次后仍无法审查 batch: {last_error}")


def write_issue(outfile: Any, issue: dict[str, Any]) -> None:
    outfile.write(json.dumps(issue, ensure_ascii=False, separators=(",", ":")) + "\n")
    outfile.flush()


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，如需替换请传入 --overwrite：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_groups = load_template_groups(
        args.templates,
        locale=args.locale,
        persona=args.persona,
        all_personas=args.all_personas,
        include_disabled=args.include_disabled,
    )
    template_counts = {key: len(value) for key, value in sorted(template_groups.items())}
    print(f"已加载模板：{template_counts}")

    pairs = iter_review_pairs(
        iter_processed_records(args.input),
        template_groups,
        max_records=args.max_records,
        max_pairs=args.max_pairs,
    )

    reviewed_pairs = 0
    issue_count = 0
    issue_counts_by_type: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as outfile:
        for batch in batch_items(pairs, args.batch_size):
            reviews = review_batch_with_retries(
                batch=batch,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
                retry_wait=args.retry_wait,
            )

            for index, item in enumerate(batch):
                reviewed_pairs += 1
                review = reviews[index]
                if review["suitable"]:
                    continue

                issue = {
                    "source_path": item["source_path"],
                    "line_no": item["line_no"],
                    "query": item["query"],
                    "filler_type": item["filler_type"],
                    "persona": item["persona"],
                    "template_id": item["template_id"],
                    "text": item["text"],
                    "reason": review["reason"],
                    "review_method": "llm_openai_compatible",
                    "review_model": args.model,
                }
                write_issue(outfile, issue)
                issue_count += 1
                issue_counts_by_type[item["filler_type"]] += 1

            if reviewed_pairs % PROGRESS_LOG_INTERVAL == 0:
                print(f"已审查 {reviewed_pairs} 组 pair；发现 {issue_count} 个问题")

    print(f"已审查 pair 数={reviewed_pairs}")
    print(f"问题数={issue_count}")
    print(f"按类型统计的问题数={dict(sorted(issue_counts_by_type.items()))}")
    print(f"不合适的 pair 已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
