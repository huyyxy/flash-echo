"""Download and pin the base transformer model used by training.

默认把 ``hfl/rbt3`` 下载到 ``models/pretrained/hfl-rbt3``，之后训练脚本可通过
``--base-model models/pretrained/hfl-rbt3`` 使用本地副本，减少重复联网和版本漂移。

使用示例::

    pip3 install -e ".[ml]"
    python3 scripts/download_pretrained_model.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoConfig, AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pretrained model for local training.")
    parser.add_argument("--model", default="hfl/rbt3", help="Hugging Face model id or local path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/pretrained/hfl-rbt3"),
        help="Directory where the downloaded model files are saved.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the remote repository.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in output-dir when it already exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}. "
            "Use --overwrite to refresh the local copy."
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading model={args.model} revision={args.revision or 'default'}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    manifest = {
        "source_model": args.model,
        "revision": args.revision,
        "saved_at_unix": int(time.time()),
        "files": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved pretrained model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
