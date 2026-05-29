"""Download MiniMind3 pretrained weights for local SFT/LoRA fine-tuning.

默认把 Hugging Face 上的 ``jingyaogong/minimind-3``（Transformers / Safetensors 格式）
下载到 ``models/pretrained/jingyaogong-minimind-3``，供后续 MiniMind3 短垫话前缀 SFT/LoRA
微调作为基础模型。

也可选下载 MiniMind 官方 PyTorch 原生 ``.pth`` 权重（来自 ``jingyaogong/minimind-3-pytorch``）。

使用示例::

    pip3 install -e ".[train]"
    python3 tools/training/download_minimind3_pretrained.py

    # 国内可用 ModelScope 镜像
    pip3 install modelscope
    python3 tools/training/download_minimind3_pretrained.py --source modelscope

    # 下载 PyTorch 原生预训练 checkpoint（非 Transformers 格式）
    python3 tools/training/download_minimind3_pretrained.py \\
      --format pytorch \\
      --weight pretrain_768 \\
      --output-dir models/pretrained/minimind-3-pytorch

    # 使用 Hugging Face 镜像（环境变量）
    HF_ENDPOINT=https://hf-mirror.com python3 tools/training/download_minimind3_pretrained.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

DEFAULT_HF_MODEL = "jingyaogong/minimind-3"
DEFAULT_MS_MODEL = "gongjy/minimind-3"
DEFAULT_HF_PYTORCH_REPO = "jingyaogong/minimind-3-pytorch"
DEFAULT_MS_PYTORCH_REPO = "gongjy/minimind-3-pytorch"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSFORMERS_OUTPUT = PROJECT_ROOT / "models/pretrained/jingyaogong-minimind-3"
DEFAULT_PYTORCH_OUTPUT = PROJECT_ROOT / "models/pretrained/minimind-3-pytorch"
DEFAULT_PYTORCH_WEIGHT = "pretrain_768"

PYTORCH_WEIGHT_CHOICES = (
    "pretrain_768",
    "pretrain_zero_768",
    "pretrain_768_moe",
    "full_sft_768",
    "full_sft_zero_768",
    "full_sft_768_moe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MiniMind3 pretrained model files.")
    parser.add_argument(
        "--format",
        choices=("transformers", "pytorch"),
        default="transformers",
        help="transformers: Safetensors + tokenizer; pytorch: native .pth checkpoint.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="Download source. modelscope requires `pip install modelscope`.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Remote model id. Defaults to jingyaogong/minimind-3 (HF) or gongjy/minimind-3 (ModelScope) "
            "for transformers format; minimind-3-pytorch repo for pytorch format."
        ),
    )
    parser.add_argument(
        "--weight",
        default=DEFAULT_PYTORCH_WEIGHT,
        choices=PYTORCH_WEIGHT_CHOICES,
        help="PyTorch checkpoint stem (without .pth). Only used when --format=pytorch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory for downloaded files.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision, branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional hub cache directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in output-dir when it already exists.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip post-download config/tokenizer validation.",
    )
    return parser.parse_args()


def resolve_remote_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.format == "pytorch":
        return DEFAULT_MS_PYTORCH_REPO if args.source == "modelscope" else DEFAULT_HF_PYTORCH_REPO
    return DEFAULT_MS_MODEL if args.source == "modelscope" else DEFAULT_HF_MODEL


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.format == "pytorch":
        return DEFAULT_PYTORCH_OUTPUT
    return DEFAULT_TRANSFORMERS_OUTPUT


def validate_args(args: argparse.Namespace) -> None:
    output_dir = resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}. "
            "Use --overwrite to refresh the local copy."
        )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def download_from_huggingface(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str | None,
    cache_dir: Path | None,
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(output_dir),
    )


def download_pytorch_weight_from_huggingface(
    *,
    repo_id: str,
    weight: str,
    output_dir: Path,
    revision: str | None,
    cache_dir: Path | None,
) -> Path:
    from huggingface_hub import hf_hub_download

    filename = f"{weight}.pth"
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    target = output_dir / filename
    shutil.copy2(downloaded, target)
    return target


def download_from_modelscope(
    *,
    model_id: str,
    output_dir: Path,
    revision: str | None,
    cache_dir: Path | None,
) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "ModelScope download requires the modelscope package. Install it with: pip install modelscope"
        ) from exc

    snapshot_download(
        model_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(output_dir),
    )


def download_pytorch_weight_from_modelscope(
    *,
    model_id: str,
    weight: str,
    output_dir: Path,
    revision: str | None,
    cache_dir: Path | None,
) -> Path:
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:
        raise ImportError(
            "ModelScope download requires the modelscope package. Install it with: pip install modelscope"
        ) from exc

    filename = f"{weight}.pth"
    downloaded = model_file_download(
        model_id,
        filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    target = output_dir / filename
    shutil.copy2(downloaded, target)
    return target


def validate_transformers_bundle(output_dir: Path) -> dict[str, object]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "tokenizer_class": tokenizer.__class__.__name__,
    }


def write_manifest(
    *,
    output_dir: Path,
    source: str,
    remote_model: str,
    revision: str | None,
    download_format: str,
    weight: str | None,
    validation: dict[str, object] | None,
) -> None:
    manifest: dict[str, object] = {
        "source": source,
        "remote_model": remote_model,
        "revision": revision,
        "format": download_format,
        "saved_at_unix": int(time.time()),
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    if weight is not None:
        manifest["pytorch_weight"] = weight
    if validation is not None:
        manifest["validation"] = validation
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir = resolve_output_dir(args)
    remote_model = resolve_remote_model(args)
    prepare_output_dir(output_dir, args.overwrite)

    print(
        f"downloading format={args.format} source={args.source} "
        f"model={remote_model} revision={args.revision or 'default'}"
    )

    validation: dict[str, object] | None = None
    weight: str | None = None

    if args.format == "transformers":
        if args.source == "huggingface":
            download_from_huggingface(
                repo_id=remote_model,
                output_dir=output_dir,
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
        else:
            download_from_modelscope(
                model_id=remote_model,
                output_dir=output_dir,
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
        if not args.skip_validate:
            validation = validate_transformers_bundle(output_dir)
            print(
                "validated transformers bundle: "
                f"model_type={validation['model_type']} "
                f"architectures={validation['architectures']}"
            )
    else:
        weight = args.weight
        if args.source == "huggingface":
            target = download_pytorch_weight_from_huggingface(
                repo_id=remote_model,
                weight=weight,
                output_dir=output_dir,
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
        else:
            target = download_pytorch_weight_from_modelscope(
                model_id=remote_model,
                weight=weight,
                output_dir=output_dir,
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
        print(f"saved pytorch checkpoint to {target}")

    write_manifest(
        output_dir=output_dir,
        source=args.source,
        remote_model=remote_model,
        revision=args.revision,
        download_format=args.format,
        weight=weight,
        validation=validation,
    )
    print(f"saved MiniMind3 pretrained files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
