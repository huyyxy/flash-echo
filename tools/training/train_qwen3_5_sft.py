"""Fine-tune Qwen3.5-0.8B on filler-prefix SFT data.

This script reuses the generic causal-LM SFT implementation from
``train_minimind3_sft.py`` while providing Qwen-specific default paths and
model version names.

Example::

    python3 tools/training/download_qwen3_5_pretrained.py
    python3 tools/training/train_qwen3_5_sft.py --persona male_white_collar

The default data source is ``data/filler_prefix/<persona>/{train,valid,test}.jsonl``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import train_minimind3_sft as sft


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED = PROJECT_ROOT / "models/pretrained/Qwen-Qwen3.5-0.8B"
DEFAULT_FILLER_DATA_ROOT = PROJECT_ROOT / "data/filler_prefix"
DEFAULT_PERSONA = "male_white_collar"


def default_base_model() -> str:
    if DEFAULT_PRETRAINED.exists():
        return str(DEFAULT_PRETRAINED)
    return "Qwen/Qwen3.5-0.8B"


def resolve_paths(args: argparse.Namespace) -> None:
    persona_dir = DEFAULT_FILLER_DATA_ROOT / args.persona
    if args.train is None:
        args.train = persona_dir / "train.jsonl"
    if args.valid is None:
        args.valid = persona_dir / "valid.jsonl"
    if args.test is None:
        args.test = persona_dir / "test.jsonl"
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / f"models/checkpoints/qwen3_5_0_8b-filler-{args.persona}-v1.0.0"
    if args.model_version is None:
        args.model_version = f"qwen3_5_0_8b-filler-{args.persona}-v1.0.0"


def parse_args() -> argparse.Namespace:
    parser = sft.parse_args()
    parser.persona = parser.persona or DEFAULT_PERSONA
    return parser


def main() -> int:
    sft.DEFAULT_PRETRAINED = DEFAULT_PRETRAINED
    sft.DEFAULT_FILLER_DATA_ROOT = DEFAULT_FILLER_DATA_ROOT
    sft.DEFAULT_PERSONA = DEFAULT_PERSONA
    sft.default_base_model = default_base_model
    sft.resolve_paths = resolve_paths
    return sft.main()


if __name__ == "__main__":
    raise SystemExit(main())
