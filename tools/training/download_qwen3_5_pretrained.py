"""Download Qwen3.5-0.8B pretrained weights from Hugging Face.

默认把 ``Qwen/Qwen3.5-0.8B`` 下载到 ``models/pretrained/Qwen-Qwen3.5-0.8B``，
供本地 SFT/LoRA 微调或推理使用。

使用示例::

    pip3 install -e ".[train]"
    python3 tools/training/download_qwen3_5_pretrained.py

    # 指定 revision 或输出目录
    python3 tools/training/download_qwen3_5_pretrained.py \\
      --revision main \\
      --output-dir models/pretrained/Qwen-Qwen3.5-0.8B

    # 使用 Hugging Face 镜像（环境变量）
    HF_ENDPOINT=https://hf-mirror.com python3 tools/training/download_qwen3_5_pretrained.py

脚本会优先使用新版 ``huggingface_hub`` 的 ``local_dir`` 直写下载；若当前环境
未安装 ``huggingface_hub`` 或版本过旧不支持 ``local_dir``，则自动回退到
旧版 hub cache 下载并复制到目标目录。

下载完成后会做文件完整性校验；若当前 ``transformers`` 版本不支持
``qwen3_5``（需 ``transformers>=5.3``），校验会跳过 ``AutoConfig`` 加载并
打印提示，不会导致脚本失败。
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

DEFAULT_HF_MODEL = "Qwen/Qwen3.5-0.8B"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "models/pretrained/Qwen-Qwen3.5-0.8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen3.5-0.8B from Hugging Face.")
    parser.add_argument(
        "--model",
        default=DEFAULT_HF_MODEL,
        help=f"Hugging Face repo id (default: {DEFAULT_HF_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Local directory for downloaded files (default: {DEFAULT_OUTPUT}).",
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


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}. "
            "Use --overwrite to refresh the local copy."
        )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download, "huggingface_hub"
    except ImportError:
        from transformers.utils.hub import snapshot_download

        return snapshot_download, "transformers.utils.hub"


def supports_local_dir(snapshot_download) -> bool:
    return "local_dir" in inspect.signature(snapshot_download).parameters


def build_snapshot_kwargs(
    *,
    repo_id: str,
    revision: str | None,
    cache_dir: Path | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"repo_id": repo_id}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def copy_snapshot_to_output_dir(source_dir: Path, output_dir: Path) -> None:
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            shutil.copy2(path.resolve(), target)
        else:
            shutil.copy2(path, target)


def download_via_local_dir(
    snapshot_download,
    *,
    kwargs: dict[str, object],
    output_dir: Path,
) -> None:
    snapshot_download(**kwargs, local_dir=str(output_dir))


def download_via_hub_cache(
    snapshot_download,
    *,
    kwargs: dict[str, object],
    output_dir: Path,
) -> None:
    cached_dir = Path(snapshot_download(**kwargs))
    copy_snapshot_to_output_dir(cached_dir, output_dir)


def download_from_huggingface(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str | None,
    cache_dir: Path | None,
) -> None:
    snapshot_download, import_source = import_snapshot_download()
    kwargs = build_snapshot_kwargs(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
    )

    if supports_local_dir(snapshot_download):
        try:
            download_via_local_dir(
                snapshot_download,
                kwargs=kwargs,
                output_dir=output_dir,
            )
        except TypeError as exc:
            if "local_dir" not in str(exc):
                raise
            print(
                "local_dir is not supported by the installed huggingface_hub; "
                "falling back to hub cache download"
            )
        else:
            print(f"downloaded via huggingface_hub local_dir API (import: {import_source})")
            return
    else:
        print(
            "installed huggingface_hub does not expose local_dir; "
            "falling back to hub cache download"
        )

    download_via_hub_cache(
        snapshot_download,
        kwargs=kwargs,
        output_dir=output_dir,
    )
    print(f"downloaded via legacy hub cache API (import: {import_source})")


def validate_download_bundle(output_dir: Path) -> dict[str, object]:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {output_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_files = sorted(path.name for path in output_dir.glob("*.safetensors"))
    has_pytorch_bin = (output_dir / "pytorch_model.bin").is_file()
    if not weight_files and not has_pytorch_bin:
        raise FileNotFoundError(f"no model weight files found in {output_dir}")

    text_config = config.get("text_config") or {}
    validation: dict[str, object] = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "vocab_size": text_config.get("vocab_size", config.get("vocab_size")),
        "weight_files": weight_files,
    }

    try:
        from transformers import AutoConfig, AutoTokenizer

        loaded_config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        validation["transformers"] = "ok"
        validation["model_type"] = getattr(loaded_config, "model_type", validation["model_type"])
        validation["architectures"] = getattr(loaded_config, "architectures", validation["architectures"])
        validation["vocab_size"] = getattr(loaded_config, "vocab_size", validation["vocab_size"])
        validation["tokenizer_class"] = tokenizer.__class__.__name__
    except (ImportError, ValueError, KeyError, OSError) as exc:
        validation["transformers"] = "skipped"
        validation["transformers_note"] = (
            f"{exc}. Qwen3.5 requires transformers>=5.3 for AutoConfig; "
            "the downloaded files are still usable after upgrading transformers."
        )
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
            validation["tokenizer_class"] = tokenizer.__class__.__name__
        except (ImportError, ValueError, KeyError, OSError) as tok_exc:
            validation["tokenizer_class"] = f"skipped: {tok_exc}"

    return validation


def write_manifest(
    *,
    output_dir: Path,
    remote_model: str,
    revision: str | None,
    validation: dict[str, object] | None,
) -> None:
    manifest: dict[str, object] = {
        "source": "huggingface",
        "remote_model": remote_model,
        "revision": revision,
        "saved_at_unix": int(time.time()),
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    if validation is not None:
        manifest["validation"] = validation
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    prepare_output_dir(args.output_dir, args.overwrite)

    print(
        f"downloading model={args.model} revision={args.revision or 'default'} "
        f"-> {args.output_dir}"
    )
    download_from_huggingface(
        repo_id=args.model,
        output_dir=args.output_dir,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    validation: dict[str, object] | None = None
    if not args.skip_validate:
        validation = validate_download_bundle(args.output_dir)
        print(
            "validated download bundle: "
            f"model_type={validation['model_type']} "
            f"architectures={validation['architectures']} "
            f"weight_files={len(validation.get('weight_files', []))}"
        )
        if validation.get("transformers") == "skipped":
            print(f"warning: {validation['transformers_note']}")

    write_manifest(
        output_dir=args.output_dir,
        remote_model=args.model,
        revision=args.revision,
        validation=validation,
    )
    print(f"saved Qwen3.5 pretrained files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
