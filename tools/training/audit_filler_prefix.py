"""使用 LLM 审核 ``data/filler_prefix`` 中 user/assistant 对话是否符合垫话规范。

规则与 ``prepare_sharegpt_filler_prefix.py`` 中 ``build_prefix_prompt`` 一致：
先做本地格式校验（长度、句末标点等），再批量调用 OpenAI 兼容 API，
结合 ``user_content`` 与 ``filler_prefix`` 判断垫话是否适合该用户话轮（不能孤立只审 assistant）。
不通过的样本从 ``train.jsonl`` / ``valid.jsonl`` / ``test.jsonl`` 移除，并追加写入同目录下的
``error.jsonl``（保留原记录并附带 ``audit`` 元数据）。每批 LLM 审核结束后会立即 flush
``error.jsonl`` 与临时通过文件，无需等整个 split 跑完。默认按 test → valid → train 顺序处理。

环境变量（与 ``prepare_sharegpt_filler_prefix.py`` 相同）::

    OPENAI_API_KEY
    OPENAI_BASE_URL
    OPENAI_MODEL

使用示例::

    # 审核 data/filler_prefix/male_white_collar/ 下三个划分文件
    python3 tools/training/audit_filler_prefix.py \\
      --data-dir data/filler_prefix/male_white_collar

    # 仅本地规则校验，不调用 LLM
    python3 tools/training/audit_filler_prefix.py --skip-llm

    # 试跑：只处理每个文件前 100 条
    python3 tools/training/audit_filler_prefix.py --max-records 100 --dry-run

    # 审核 data/filler_prefix 下所有子目录
    python3 tools/training/audit_filler_prefix.py --data-dir data/filler_prefix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Sequence, TextIO

from prepare_sharegpt_filler_prefix import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_WORKERS,
    MAX_PREFIX_CHARS,
    MIN_PREFIX_CHARS,
    PERSONA_DESCRIPTIONS,
    PREFERRED_MAX_PREFIX_CHARS,
    PROJECT_ROOT,
    RETRYABLE_EXCEPTIONS,
    VALID_PREFIX_SUFFIXES,
    call_openai_compatible_chat,
    config_value,
    is_user_turn,
    load_dotenv,
    normalize_filler_prefix,
    normalize_text,
    parse_json_object,
    parse_prefix_id,
    turn_content,
    validate_filler_prefix,
)

DEFAULT_SPLITS = ("train", "valid", "test")
# 先处理小文件，便于尽早看到 error.jsonl 与进度；train 通常最大放最后。
SPLIT_PROCESS_ORDER = ("test", "valid", "train")
GENERIC_PERSONA_DESCRIPTION = "通用中文口语助手，语气自然、礼貌，不夸张。"

SPLIT_FILENAMES = {split: f"{split}.jsonl" for split in DEFAULT_SPLITS}
ERROR_FILENAME = "error.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    dotenv_values = load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Audit filler-prefix assistant content in data/filler_prefix JSONL splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/filler_prefix"),
        help=(
            "Persona output directory (contains train/valid/test.jsonl) or parent directory "
            "whose immediate subdirectories are processed."
        ),
    )
    parser.add_argument(
        "--persona",
        choices=sorted(PERSONA_DESCRIPTIONS),
        default=None,
        help="Persona for style checks. Defaults to the leaf directory name when recognized.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(DEFAULT_SPLITS),
        default=list(DEFAULT_SPLITS),
        help="Which split files to audit.",
    )
    parser.add_argument(
        "--api-key",
        default=config_value("OPENAI_API_KEY", dotenv_values=dotenv_values),
    )
    parser.add_argument(
        "--base-url",
        default=config_value(
            "OPENAI_BASE_URL",
            dotenv_values=dotenv_values,
            default=DEFAULT_OPENAI_BASE_URL,
        ),
    )
    parser.add_argument(
        "--model",
        default=config_value("OPENAI_MODEL", dotenv_values=dotenv_values),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Max records per split file. 0 means no limit.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only run local validate_filler_prefix; do not call the API.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stats without rewriting split files or appending error.jsonl.",
    )
    parser.add_argument(
        "--overwrite-error",
        action="store_true",
        help="Truncate error.jsonl in each processed directory before auditing.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print progress every N audited records per directory. 0 disables.",
    )
    return parser.parse_args(argv)


def persona_for_directory(data_dir: Path, explicit: str | None) -> tuple[str, str]:
    if explicit is not None:
        return explicit, PERSONA_DESCRIPTIONS[explicit]
    name = data_dir.name
    if name in PERSONA_DESCRIPTIONS:
        return name, PERSONA_DESCRIPTIONS[name]
    return name, GENERIC_PERSONA_DESCRIPTION


def build_audit_prompt(
    items: list[tuple[int, str, str]],
    *,
    persona: str,
    persona_description: str,
) -> str:
    payload = [
        {"id": index, "user_content": user_content, "filler_prefix": filler_prefix}
        for index, (_, user_content, filler_prefix) in enumerate(items)
    ]
    suffix_examples = "、".join(repr(s) for s in VALID_PREFIX_SUFFIXES[:6])

    return (
        "你是实时语音交互系统的数据质检助手。每条样本包含 user_content（用户轮次原文）"
        "与 filler_prefix（assistant 轮次的短垫话前缀）。\n\n"

        "审核方法（必须遵守）：\n"
        "1. 先完整阅读 user_content，理解用户这句话的话题、意图与语气。\n"
        "2. 再判断 filler_prefix 是否适合作为对该 user_content 的垫话承接——"
        "不能孤立地只审 filler_prefix。\n"
        "3. filler_prefix 是用户说完话后、主模型正式回答前播放的一句自然承接语，"
        "不是对用户问题的正式回答。\n"
        "4. 若 user_content 含有「用户：」「请生成一句可续写的短垫话前缀」等训练模板，"
        "应依据模板前面的真实用户问题判断，不要被模板字面误导。\n\n"

        f"Persona:\n{persona}，{persona_description}\n\n"

        "审核标准（与数据合成规范一致）：\n"
        "1. 必须是一句短前缀，不能是多句解释或 Markdown。\n"
        f"2. 实义字数量应在 {MIN_PREFIX_CHARS} 到 {MAX_PREFIX_CHARS} 之间；"
        f"优先 {MIN_PREFIX_CHARS} 到 {PREFERRED_MAX_PREFIX_CHARS} 字。\n"
        "3. 自然、口语化，适合 TTS 立即播放。\n"
        "4. filler_prefix 须与 user_content 的话题、意图相匹配，不能答非所问或套用无关万能模板。\n"
        "5. 有衔接感：针对该句 user_content，主模型能从 filler_prefix 后自然续写完整回答。\n"
        "6. 不能是未说完的名词短语或残缺句（如「昨天的电影是」「你的问题主要是」）。\n"
        f"7. 句末必须以逗号、句号、感叹号或问号等结尾（如 {suffix_examples} 等）。\n"
        "8. 不得直接回答 user_content 中的问题，不得给出方案、结论、原因、步骤或建议。\n"
        "9. 不得编造 user_content 未提及的事实、数据、时间、地点、人物、状态或外部信息。\n"
        "10. 不得声称已完成的操作（如「我查到了」「已经帮你处理好了」）。\n"
        "11. 不得做结果承诺（如「一定帮你解决」「肯定没问题」）。\n"
        "12. 不得用改变对话方向的反问（如「你是想问这个吗？」）。\n"
        "13. 语气符合 Persona，且与 user_content 的语境协调。\n"
        "14. 若 pass 为 false，reason 用一句中文说明问题，并点明与 user_content 的关系"
        "（如「用户问天气，垫话却谈订单」）。\n"
        "15. 必须为每个输入 id 输出且只输出一条审核结果，id 与输入一致。\n"
        "16. 只输出 JSON 对象，不要 Markdown 或解释性文字。\n\n"

        "语境不匹配坏例子：\n"
        'user_content="今天天气怎么样？" filler_prefix="关于你这笔订单的状态,"\n'
        'user_content="帮我写请假邮件" filler_prefix="宇宙奥秘很有意思,"\n\n'

        "输出格式：\n"
        '{"reviews":[{"id":0,"pass":true,"reason":""},{"id":1,"pass":false,'
        '"reason":"垫话与用户问题无关"}]}\n\n'

        "待审核样本：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_llm_review_map(response_text: str) -> dict[int, tuple[bool, str]]:
    parsed = parse_json_object(response_text)
    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("LLM response must contain a reviews list")

    result: dict[int, tuple[bool, str]] = {}
    for item in reviews:
        if not isinstance(item, dict):
            raise ValueError("each review item must be an object")
        review_id = parse_prefix_id(item.get("id"))
        if review_id in result:
            continue
        passed = item.get("pass")
        if not isinstance(passed, bool):
            raise ValueError(f"pass for id={review_id} must be a boolean")
        reason = item.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError(f"reason for id={review_id} must be a string")
        result[review_id] = (passed, reason.strip())
    return result


def call_audit_batch(
    batch: list[AuditItem],
    *,
    persona: str,
    persona_description: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
) -> str:
    payload = [
        (index, item.user_content, item.filler_prefix) for index, item in enumerate(batch)
    ]
    return call_openai_compatible_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=build_audit_prompt(payload, persona=persona, persona_description=persona_description),
        temperature=temperature,
        timeout=timeout,
    )


def audit_batch_with_retries(
    batch: list[AuditItem],
    *,
    persona: str,
    persona_description: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    retries: int,
    retry_wait: float,
) -> tuple[dict[int, tuple[bool, str]], dict[int, str]]:
    """Return (reviews by batch index, errors by batch index)."""
    pending = list(range(len(batch)))
    results: dict[int, tuple[bool, str]] = {}
    errors: dict[int, str] = {}

    while pending and retries > 0:
        sub_batch = [batch[index] for index in pending]
        try:
            response_text = call_audit_batch(
                sub_batch,
                persona=persona,
                persona_description=persona_description,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout=timeout,
            )
            parsed = parse_llm_review_map(response_text)
        except RETRYABLE_EXCEPTIONS as exc:
            retries -= 1
            if len(pending) > 1:
                mid = len(pending) // 2
                left, right = pending[:mid], pending[mid:]
                left_results, left_errors = audit_batch_with_retries(
                    [batch[i] for i in left],
                    persona=persona,
                    persona_description=persona_description,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                    retries=retries,
                    retry_wait=retry_wait,
                )
                right_results, right_errors = audit_batch_with_retries(
                    [batch[i] for i in right],
                    persona=persona,
                    persona_description=persona_description,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                    retries=retries,
                    retry_wait=retry_wait,
                )
                for local_idx, review in left_results.items():
                    results[left[local_idx]] = review
                for local_idx, review in right_results.items():
                    results[right[local_idx]] = review
                for local_idx, err in left_errors.items():
                    errors[left[local_idx]] = err
                for local_idx, err in right_errors.items():
                    errors[right[local_idx]] = err
                return results, errors
            errors[pending[0]] = str(exc)
            if retries > 0:
                time.sleep(retry_wait)
            continue

        missing: list[int] = []
        for local_index, original_index in enumerate(pending):
            review = parsed.get(local_index)
            if review is None:
                missing.append(original_index)
                errors[original_index] = f"missing review id={local_index}"
                continue
            results[original_index] = review

        if not missing:
            return results, errors

        if len(missing) == 1 and retries > 1:
            pending = missing
            retries -= 1
            time.sleep(retry_wait)
            continue
        for index in missing:
            if index not in results:
                errors[index] = errors.get(index, "missing LLM review")
        return results, errors

    for index in pending:
        if index not in results:
            errors[index] = errors.get(index, "audit retries exhausted")
    return results, errors


class AuditItem:
    __slots__ = ("line_no", "record", "user_content", "filler_prefix")

    def __init__(
        self,
        *,
        line_no: int,
        record: dict[str, Any],
        user_content: str,
        filler_prefix: str,
    ) -> None:
        self.line_no = line_no
        self.record = record
        self.user_content = user_content
        self.filler_prefix = filler_prefix


def load_split_records(path: Path, *, max_records: int) -> list[AuditItem]:
    items: list[AuditItem] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            if max_records > 0 and len(items) >= max_records:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"skip invalid json at {path}:{line_no}: {exc}", file=sys.stderr)
                continue

            user_content = user_content_from_record(record)
            filler_prefix = assistant_content_from_record(record)
            if user_content is None or filler_prefix is None:
                print(
                    f"skip malformed record at {path}:{line_no}",
                    file=sys.stderr,
                )
                continue
            items.append(
                AuditItem(
                    line_no=line_no,
                    record=record,
                    user_content=user_content,
                    filler_prefix=normalize_filler_prefix(filler_prefix),
                )
            )
    return items


def user_content_from_record(record: dict[str, Any]) -> str | None:
    """Return the first user turn content exactly as stored (whitespace-normalized)."""
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if not isinstance(turn, dict) or not is_user_turn(turn):
            continue
        content = turn_content(turn)
        if isinstance(content, str) and content.strip():
            return normalize_text(content)
    return None


def assistant_content_from_record(record: dict[str, Any]) -> str | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if isinstance(role, str) and role.lower() == "assistant":
            content = turn.get("content")
            if isinstance(content, str):
                return content
    return None


def local_audit(item: AuditItem) -> str | None:
    try:
        validate_filler_prefix(item.filler_prefix)
    except ValueError as exc:
        return str(exc)
    return None


def enrich_error_record(
    record: dict[str, Any],
    *,
    reason: str,
    stage: str,
    source_file: str,
    line_no: int,
) -> dict[str, Any]:
    output = dict(record)
    output["audit"] = {
        "pass": False,
        "reason": reason,
        "stage": stage,
        "source_file": source_file,
        "line_no": line_no,
    }
    return output


def append_jsonl(outfile: TextIO, records: Sequence[dict[str, Any]]) -> None:
    for record in records:
        outfile.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    outfile.flush()


def ordered_splits(splits: Sequence[str]) -> list[str]:
    order_rank = {name: index for index, name in enumerate(SPLIT_PROCESS_ORDER)}
    return sorted(splits, key=lambda name: order_rank.get(name, len(SPLIT_PROCESS_ORDER)))


def discover_data_dirs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    if any((data_dir / SPLIT_FILENAMES[split]).exists() for split in DEFAULT_SPLITS):
        return [data_dir]
    subdirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    result = [
        subdir
        for subdir in subdirs
        if any((subdir / SPLIT_FILENAMES[split]).exists() for split in DEFAULT_SPLITS)
    ]
    if not result:
        raise FileNotFoundError(
            f"no split jsonl files found under {data_dir} or its immediate subdirectories"
        )
    return result


def audit_split_file(
    path: Path,
    *,
    persona: str,
    persona_description: str,
    args: argparse.Namespace,
    error_outfile: TextIO | None,
    counters: Counter[str],
) -> None:
    items = load_split_records(path, max_records=args.max_records)
    if not items:
        print(f"skip empty or missing records: {path}", file=sys.stderr)
        return

    passed_count = 0
    error_count = 0
    llm_pending: list[AuditItem] = []
    passed_tmp = path.with_name(path.name + ".audit.tmp")
    passed_out: TextIO | None = None

    if not args.dry_run:
        passed_out = passed_tmp.open("w", encoding="utf-8")

    def flush_passed(records: Sequence[dict[str, Any]]) -> None:
        nonlocal passed_count
        if not records:
            return
        passed_count += len(records)
        if passed_out is not None:
            append_jsonl(passed_out, records)

    def flush_errors(records: Sequence[dict[str, Any]]) -> None:
        nonlocal error_count
        if not records:
            return
        error_count += len(records)
        if error_outfile is not None:
            append_jsonl(error_outfile, records)

    for item in items:
        local_reason = local_audit(item)
        if local_reason is not None:
            counters["local_fail"] += 1
            flush_errors(
                [
                    enrich_error_record(
                        item.record,
                        reason=local_reason,
                        stage="local",
                        source_file=path.name,
                        line_no=item.line_no,
                    )
                ]
            )
            continue
        if args.skip_llm:
            counters["pass"] += 1
            flush_passed([item.record])
            continue
        llm_pending.append(item)

    if not args.skip_llm and llm_pending:
        batch_size = args.batch_size
        batches: list[list[AuditItem]] = []
        batch: list[AuditItem] = []
        for item in llm_pending:
            batch.append(item)
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)

        processed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    audit_batch_with_retries,
                    current_batch,
                    persona=persona,
                    persona_description=persona_description,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_wait=args.retry_wait,
                ): current_batch
                for current_batch in batches
            }
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    current_batch = futures.pop(future)
                    batch_passed: list[dict[str, Any]] = []
                    batch_errors: list[dict[str, Any]] = []
                    try:
                        reviews, batch_api_errors = future.result()
                    except Exception as exc:
                        print(
                            f"batch audit failed for {path}: {exc}",
                            file=sys.stderr,
                        )
                        for item in current_batch:
                            counters["api_fail_kept"] += 1
                            batch_passed.append(item.record)
                        flush_passed(batch_passed)
                        continue

                    for index, item in enumerate(current_batch):
                        processed += 1
                        if index in batch_api_errors and index not in reviews:
                            counters["api_fail_kept"] += 1
                            print(
                                f"keep record after API failure {path}:{item.line_no}: "
                                f"{batch_api_errors[index]}",
                                file=sys.stderr,
                            )
                            batch_passed.append(item.record)
                            continue
                        review = reviews.get(index)
                        if review is None:
                            counters["api_fail_kept"] += 1
                            batch_passed.append(item.record)
                            continue
                        passed, reason = review
                        if passed:
                            counters["pass"] += 1
                            batch_passed.append(item.record)
                        else:
                            counters["llm_fail"] += 1
                            batch_errors.append(
                                enrich_error_record(
                                    item.record,
                                    reason=reason or "未通过垫话规范审核",
                                    stage="llm",
                                    source_file=path.name,
                                    line_no=item.line_no,
                                )
                            )

                    flush_passed(batch_passed)
                    flush_errors(batch_errors)

                    if args.log_interval > 0 and processed % args.log_interval == 0:
                        print(
                            f"{path}: audited {processed}/{len(llm_pending)} llm samples, "
                            f"errors_so_far={error_count} (local_fail={counters['local_fail']}, "
                            f"llm_fail={counters['llm_fail']})",
                            file=sys.stderr,
                        )

    if args.dry_run:
        print(
            f"[dry-run] {path}: total={len(items)} pass={passed_count} errors={error_count}",
            file=sys.stderr,
        )
        return

    if passed_out is not None:
        passed_out.close()
        os.replace(passed_tmp, path)
    elif passed_tmp.exists():
        passed_tmp.unlink()

    print(
        f"{path}: kept {passed_count}, moved {error_count} to {ERROR_FILENAME}",
        file=sys.stderr,
    )


def validate_audit_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError("workers must be greater than 0")
    if args.retries <= 0:
        raise ValueError("retries must be greater than 0")
    if args.retry_wait < 0:
        raise ValueError("retry-wait must be non-negative")
    if args.timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    if args.max_records < 0:
        raise ValueError("max-records must be non-negative")
    if args.log_interval < 0:
        raise ValueError("log-interval must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")
    if not args.skip_llm:
        if not args.api_key:
            raise ValueError("missing API key. Set OPENAI_API_KEY or pass --api-key.")
        if not args.model:
            raise ValueError("missing model. Set OPENAI_MODEL or pass --model.")


def audit_directory(data_dir: Path, args: argparse.Namespace) -> Counter[str]:
    persona, persona_description = persona_for_directory(data_dir, args.persona)
    counters: Counter[str] = Counter()
    print(f"auditing {data_dir} (persona={persona})", file=sys.stderr)

    error_path = data_dir / ERROR_FILENAME
    error_outfile: TextIO | None = None
    if not args.dry_run:
        if args.overwrite_error:
            error_path.write_text("", encoding="utf-8")
        error_outfile = error_path.open("a", encoding="utf-8")

    try:
        for split in ordered_splits(args.splits):
            split_path = data_dir / SPLIT_FILENAMES[split]
            if not split_path.exists():
                print(f"skip missing split: {split_path}", file=sys.stderr)
                continue
            audit_split_file(
                split_path,
                persona=persona,
                persona_description=persona_description,
                args=args,
                error_outfile=error_outfile,
                counters=counters,
            )
    finally:
        if error_outfile is not None:
            error_outfile.close()

    return counters


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_audit_args(args)
        data_dirs = discover_data_dirs(args.data_dir.resolve())
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total: Counter[str] = Counter()
    for data_dir in data_dirs:
        counters = audit_directory(data_dir, args)
        total.update(counters)
        print(
            f"summary {data_dir}: {dict(counters)}",
            file=sys.stderr,
        )

    print(f"total: {dict(total)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
