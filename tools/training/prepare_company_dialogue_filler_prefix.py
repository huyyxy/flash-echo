"""将 ``data/raw/company`` 下的线上对话 JSON 日志转为 MiniMind3 SFT 训练格式。

每条原始日志包含 ``request_text``（用户输入）、``response_first_sentence``（首句 /
垫话前缀）和 ``response_text``（完整回复）。脚本抽取 ``(query, filler_prefix)`` 对，
校验后写入 ShareGPT ``conversations`` 格式的 ``train.jsonl`` / ``valid.jsonl`` /
``test.jsonl``，可直接用于 ``train_minimind3_sft.py``。

前缀来源（``--prefix-source``）：

- ``auto``（默认）：优先 ``response_first_sentence``，缺失时从 ``response_text`` 推断首句
- ``first-sentence-field``：仅使用 ``response_first_sentence`` 字段
- ``inferred``：仅从 ``response_text`` 推断首句

仅保留 ``response_text`` 在首句之后仍有内容的样本（排除完整静态短回复）。

使用示例::

    python3 tools/training/prepare_company_dialogue_filler_prefix.py

    python3 tools/training/prepare_company_dialogue_filler_prefix.py \\
      --input-dir data/raw/company \\
      --output-dir data/filler_prefix/company \\
      --prefix-source first-sentence-field

    python3 tools/training/prepare_company_dialogue_filler_prefix.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from prepare_sharegpt_filler_prefix import (
    SFT_USER_PREFIX,
    SFT_USER_PROMPT_SUFFIX,
    build_sft_record,
    build_training_user_content,
    normalize_filler_prefix,
    normalize_text,
    split_for_query,
    validate_filler_prefix,
    validate_ratios,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/raw/company"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/filler_prefix/company"
DEFAULT_SPLIT_SEED = "company_dialogue"
DEFAULT_TRAIN_RATIO = 0.96
DEFAULT_VALID_RATIO = 0.02
DEFAULT_TEST_RATIO = 0.02

PREFIX_BOUNDARY_RE = re.compile(r"[。！？!?，,…]+")
BRACKET_SEGMENT_RE = re.compile(r"【[^】]*】")
ALLOWED_STATUSES = frozenset({None, "completed"})


def remove_bracket_segments(text: str) -> str:
    """Remove action tags wrapped in full-width brackets, e.g. ``【挥手】``."""
    return normalize_text(BRACKET_SEGMENT_RE.sub("", text))


@dataclass(frozen=True)
class DialogueSample:
    query: str
    filler_prefix: str
    prefix_source: str
    group_id: str | None
    robot_id: str | None
    request_id: str | None
    source_path: str


@dataclass
class SkipStats:
    counts: Counter[str]

    def add(self, reason: str, n: int = 1) -> None:
        self.counts[reason] += n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert company dialogue JSON logs into MiniMind3 filler-prefix SFT JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing downloaded dialogue JSON logs (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for train/valid/test JSONL (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--prefix-source",
        choices=("auto", "first-sentence-field", "inferred"),
        default="auto",
        help="How to derive filler_prefix from each dialogue log.",
    )
    parser.add_argument(
        "--split-seed",
        default=DEFAULT_SPLIT_SEED,
        help=f"Stable hash seed for train/valid/test split (default: {DEFAULT_SPLIT_SEED}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Train split ratio (default: {DEFAULT_TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=DEFAULT_VALID_RATIO,
        help=f"Validation split ratio (default: {DEFAULT_VALID_RATIO}).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Test split ratio (default: {DEFAULT_TEST_RATIO}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output JSONL files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most N JSON files (useful for smoke tests).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory not found: {args.input_dir}")
    if args.max_files is not None and args.max_files < 1:
        raise SystemExit("--max-files must be >= 1")


def iter_dialogue_files(input_dir: Path) -> Iterator[Path]:
    yield from sorted(input_dir.rglob("*.json"))


def infer_prefix_from_response(response_text: str) -> str | None:
    response_text = response_text.strip()
    if not response_text:
        return None

    for match in PREFIX_BOUNDARY_RE.finditer(response_text):
        end = match.end()
        remainder = response_text[end:].strip()
        if remainder:
            return response_text[:end].strip()
    return None


def response_continues(response_text: str, prefix: str, raw_prefix: str | None) -> bool:
    response_text = response_text.strip()
    candidates = [prefix]
    if raw_prefix and str(raw_prefix).strip():
        candidates.insert(0, str(raw_prefix).strip())

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if response_text.startswith(candidate):
            return bool(response_text[len(candidate) :].strip())
    return False


def pick_prefix(
    *,
    response_text: str,
    raw_first_sentence: Any,
    prefix_source: str,
) -> tuple[str | None, str | None]:
    raw_first = str(raw_first_sentence).strip() if raw_first_sentence else ""
    normalized_first = normalize_filler_prefix(raw_first) if raw_first else ""
    inferred = infer_prefix_from_response(response_text)
    normalized_inferred = normalize_filler_prefix(inferred) if inferred else ""

    if prefix_source == "first-sentence-field":
        if not normalized_first:
            return None, None
        return normalized_first, "response_first_sentence"

    if prefix_source == "inferred":
        if not normalized_inferred:
            return None, None
        return normalized_inferred, "inferred"

    if normalized_first:
        return normalized_first, "response_first_sentence"
    if normalized_inferred:
        return normalized_inferred, "inferred"
    return None, None


def parse_dialogue_file(
    path: Path,
    *,
    prefix_source: str,
    stats: SkipStats,
) -> DialogueSample | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        stats.add("invalid_json")
        print(f"skip {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(payload, dict):
        stats.add("invalid_payload")
        return None

    status = payload.get("response_status")
    if status not in ALLOWED_STATUSES:
        stats.add("skip_status")
        return None

    query = remove_bracket_segments(str(payload.get("request_text") or ""))
    response_text = remove_bracket_segments(str(payload.get("response_text") or ""))
    if not query:
        stats.add("empty_query")
        return None
    if not response_text:
        stats.add("empty_response")
        return None

    raw_first = payload.get("response_first_sentence")
    cleaned_raw_first = (
        remove_bracket_segments(str(raw_first)) if raw_first and str(raw_first).strip() else None
    )
    prefix, source = pick_prefix(
        response_text=response_text,
        raw_first_sentence=cleaned_raw_first,
        prefix_source=prefix_source,
    )
    if not prefix or not source:
        stats.add("no_prefix")
        return None

    prefix = remove_bracket_segments(prefix)
    if not response_continues(response_text, prefix, cleaned_raw_first):
        stats.add("complete_reply")
        return None

    try:
        validate_filler_prefix(prefix)
    except ValueError as exc:
        stats.add(f"invalid_prefix:{exc.args[0]}")
        return None

    return DialogueSample(
        query=query,
        filler_prefix=prefix,
        prefix_source=source,
        group_id=_optional_str(payload.get("group_id")),
        robot_id=_optional_str(payload.get("robot_id")),
        request_id=_optional_str(payload.get("request_id")),
        source_path=str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def dedupe_samples(samples: list[DialogueSample]) -> tuple[list[DialogueSample], int]:
    best_by_query: dict[str, tuple[tuple[int, int], DialogueSample]] = {}
    duplicates = 0

    for sample in samples:
        priority = (
            0 if sample.prefix_source == "response_first_sentence" else 1,
            len(sample.filler_prefix),
        )
        existing = best_by_query.get(sample.query)
        if existing is None:
            best_by_query[sample.query] = (priority, sample)
            continue

        duplicates += 1
        if priority < existing[0]:
            best_by_query[sample.query] = (priority, sample)

    return [item[1] for item in best_by_query.values()], duplicates


def clean_user_content(content: str) -> str:
    if content.startswith(SFT_USER_PREFIX) and content.endswith(SFT_USER_PROMPT_SUFFIX):
        query = content[len(SFT_USER_PREFIX) : -len(SFT_USER_PROMPT_SUFFIX)].removesuffix("\n")
        return build_training_user_content(remove_bracket_segments(query))
    return remove_bracket_segments(content)


def clean_conversation_content(record: dict[str, Any]) -> dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return record

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            turn["content"] = clean_user_content(content)
        elif role == "assistant":
            turn["content"] = remove_bracket_segments(content)
    return record


def build_output_record(sample: DialogueSample) -> dict[str, Any]:
    query = remove_bracket_segments(sample.query)
    filler_prefix = remove_bracket_segments(sample.filler_prefix)
    record = build_sft_record(query, filler_prefix)
    record["source"] = "company_dialogue"
    record["prefix_source"] = sample.prefix_source
    if sample.group_id:
        record["group_id"] = sample.group_id
    if sample.robot_id:
        record["robot_id"] = sample.robot_id
    if sample.request_id:
        record["request_id"] = sample.request_id
    record["source_path"] = sample.source_path
    return record


def write_split(
    *,
    output_dir: Path,
    split_name: str,
    records: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    output_path = output_dir / f"{split_name}.jsonl"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False))
            outfile.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)

    stats = SkipStats(Counter())
    parsed: list[DialogueSample] = []

    for index, path in enumerate(iter_dialogue_files(args.input_dir), start=1):
        if args.max_files is not None and index > args.max_files:
            break
        sample = parse_dialogue_file(path, prefix_source=args.prefix_source, stats=stats)
        if sample is not None:
            parsed.append(sample)

    deduped, duplicate_count = dedupe_samples(parsed)

    split_buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for sample in deduped:
        split_name = split_for_query(
            sample.query,
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )
        split_buckets[split_name].append(clean_conversation_content(build_output_record(sample)))

    for split_name, records in split_buckets.items():
        write_split(
            output_dir=args.output_dir,
            split_name=split_name,
            records=records,
            overwrite=args.overwrite,
        )

    prefix_sources = Counter(sample.prefix_source for sample in deduped)
    print(
        f"input_dir={args.input_dir}\n"
        f"output_dir={args.output_dir}\n"
        f"parsed={len(parsed)} deduped={len(deduped)} duplicates_removed={duplicate_count}\n"
        f"prefix_sources={dict(prefix_sources)}\n"
        f"splits=train:{len(split_buckets['train'])} "
        f"valid:{len(split_buckets['valid'])} test:{len(split_buckets['test'])}"
    )
    if stats.counts:
        print("skipped:", dict(stats.counts.most_common(20)))
        remaining = sum(stats.counts.values()) - sum(count for _, count in stats.counts.most_common(20))
        if remaining > 0:
            print(f"skipped_other_reasons={remaining}")


if __name__ == "__main__":
    main()
