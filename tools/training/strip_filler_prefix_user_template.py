"""去掉 ``data/filler_prefix`` 数据集中 user 轮次上的 SFT 输入模板包装。

将::

    用户：{query}
    请生成一句可续写的短垫话前缀：

还原为纯 ``{query}``。已是纯 query 的记录保持不变。

使用示例::

    # 预览 data/filler_prefix 下所有 jsonl（默认跳过 error.jsonl）
    python3 tools/training/strip_filler_prefix_user_template.py --dry-run

    # 原地改写
    python3 tools/training/strip_filler_prefix_user_template.py

    # 仅处理指定子目录
    python3 tools/training/strip_filler_prefix_user_template.py \\
      --data-dir data/filler_prefix/male_white_collar
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

from prepare_sharegpt_filler_prefix import (
    SFT_USER_PREFIX,
    SFT_USER_PROMPT_SUFFIX,
    normalize_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/filler_prefix"
DEFAULT_SPLITS = ("train", "valid", "test")
ERROR_FILENAME = "error.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip SFT user template wrapper from filler_prefix JSONL datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root directory to scan (default: {DEFAULT_DATA_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--include-error",
        action="store_true",
        help=f"Also process {ERROR_FILENAME} (default: skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print stats; do not modify files",
    )
    return parser.parse_args(argv)


def iter_target_files(data_dir: Path, include_error: bool) -> Iterator[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    seen: set[Path] = set()
    for split in DEFAULT_SPLITS:
        path = data_dir / f"{split}.jsonl"
        if path.is_file():
            seen.add(path.resolve())
            yield path

    if include_error:
        path = data_dir / ERROR_FILENAME
        if path.is_file():
            seen.add(path.resolve())
            yield path

    for path in sorted(data_dir.rglob("*.jsonl")):
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not include_error and path.name == ERROR_FILENAME:
            continue
        yield path


def strip_user_content(content: str) -> tuple[str, bool]:
    if not content.startswith(SFT_USER_PREFIX):
        return content, False
    if not content.endswith(SFT_USER_PROMPT_SUFFIX):
        return content, False

    query = content[len(SFT_USER_PREFIX) : -len(SFT_USER_PROMPT_SUFFIX)].removesuffix("\n")
    return normalize_text(query), True


def transform_record(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return record, False

    changed = False
    new_conversations: list[Any] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            new_conversations.append(turn)
            continue
        if turn.get("role") != "user":
            new_conversations.append(turn)
            continue

        content = turn.get("content")
        if not isinstance(content, str):
            new_conversations.append(turn)
            continue

        stripped, did_strip = strip_user_content(content)
        if did_strip:
            changed = True
            new_conversations.append({**turn, "content": stripped})
        else:
            new_conversations.append(turn)

    if not changed:
        return record, False

    return {**record, "conversations": new_conversations}, True


def process_file(path: Path, dry_run: bool) -> Counter[str]:
    stats: Counter[str] = Counter()
    output_lines: list[str] = []

    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            raw = line.rstrip("\n")
            if not raw.strip():
                stats["blank_lines"] += 1
                output_lines.append(line.rstrip("\n"))
                continue

            stats["total"] += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                output_lines.append(raw)
                continue

            if not isinstance(record, dict):
                stats["non_object_records"] += 1
                output_lines.append(raw)
                continue

            new_record, changed = transform_record(record)
            if changed:
                stats["stripped"] += 1
                output_lines.append(json.dumps(new_record, ensure_ascii=False))
            else:
                stats["unchanged"] += 1
                output_lines.append(raw)

    if dry_run or stats["stripped"] == 0:
        return stats

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as outfile:
        for idx, line in enumerate(output_lines):
            if idx:
                outfile.write("\n")
            outfile.write(line)
    tmp_path.replace(path)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()

    try:
        target_files = list(iter_target_files(data_dir, args.include_error))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not target_files:
        print(f"No JSONL files found under {data_dir}", file=sys.stderr)
        return 1

    grand_total = Counter[str]()
    for path in target_files:
        stats = process_file(path, dry_run=args.dry_run)
        grand_total.update(stats)
        rel_path = path.relative_to(PROJECT_ROOT)
        mode = "would strip" if args.dry_run else "stripped"
        print(
            f"{rel_path}: total={stats['total']} {mode}={stats['stripped']} "
            f"unchanged={stats['unchanged']}"
        )

    print(
        f"Done ({'dry-run' if args.dry_run else 'written'}): "
        f"files={len(target_files)} total={grand_total['total']} "
        f"stripped={grand_total['stripped']} unchanged={grand_total['unchanged']}"
    )
    if grand_total["json_errors"]:
        print(f"Warning: {grand_total['json_errors']} JSON parse errors left unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
