"""将 Qwen3.5-0.8B 预训练权重导出为 ONNX 部署包。

默认读取 ``models/pretrained/Qwen-Qwen3.5-0.8B``，导出到
``models/deploy/qwen3.5-0.8b-v1.0.0/``。

Qwen3.5 使用 hybrid decoder（GatedDeltaNet + Attention），Hugging Face Optimum
目前无法稳定导出该架构。默认使用 **onnxruntime-genai** model builder；也保留
Optimum 路径供环境已对齐时使用。

部署包通常包含：

- ``model.onnx``：ONNX 推理图
- ``genai_config.json``：onnxruntime-genai 运行时配置（genai 后端）
- tokenizer / processor 文件
- ``model_card.json`` / ``export_manifest.json``：导出元数据

依赖::

    pip install 'transformers>=5.3' torch onnx onnxruntime
    pip install --pre onnxruntime-genai onnx-ir

使用示例::

    pip3 install -e ".[infer]"
    pip3 install 'transformers>=5.3' --pre onnxruntime-genai onnx-ir

    # 默认导出（onnxruntime-genai, fp32, CPU）
    python3 tools/export/export_qwen3_5_onnx.py

    # 指定本地模型与输出目录
    python3 tools/export/export_qwen3_5_onnx.py \\
      --model-dir models/pretrained/Qwen-Qwen3.5-0.8B \\
      --output-dir models/deploy/qwen3.5-0.8b-v1.0.0 \\
      --precision fp32 \\
      --execution-provider cpu \\
      --overwrite

    # 使用 Optimum（实验性，当前 Qwen3.5 通常不可用）
    python3 tools/export/export_qwen3_5_onnx.py --export-backend optimum
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models/pretrained/Qwen-Qwen3.5-0.8B"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models/deploy/qwen3.5-0.8b-v1.0.0"
DEFAULT_MODEL_VERSION = "qwen3.5-0.8b-v1.0.0"
MIN_TRANSFORMERS_VERSION = (5, 3, 0)
ORTGENAI_QWEN35_TEXT_ARCH = "Qwen3_5ForCausalLM"
ORTGENAI_QWEN35_BUILDER_ARCH = "Qwen3_5ForConditionalGeneration"


def parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qwen3.5-0.8B to ONNX.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Local Hugging Face model directory (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"ONNX bundle output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"Recorded in model_card.json (default: {DEFAULT_MODEL_VERSION}).",
    )
    parser.add_argument(
        "--export-backend",
        default="auto",
        choices=("auto", "onnxruntime-genai", "optimum"),
        help=(
            "Export backend. auto prefers onnxruntime-genai for Qwen3.5 hybrid decoder."
        ),
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=("fp32", "fp16", "bf16", "int4"),
        help="Model precision for onnxruntime-genai builder.",
    )
    parser.add_argument(
        "--execution-provider",
        default="cpu",
        choices=("cpu", "cuda", "dml", "webgpu", "NvTensorRtRtx"),
        help="Execution provider for onnxruntime-genai builder.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hub/build cache directory for onnxruntime-genai builder.",
    )
    parser.add_argument(
        "--genai-builder",
        type=Path,
        default=None,
        help=(
            "Path to onnxruntime-genai builder.py. "
            "Defaults to `python -m onnxruntime_genai.models.builder` when installed."
        ),
    )
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help=(
            "Export decoder-only bundle with inputs_embeds. "
            "Not compatible with tools/inference/chat_qwen3_5_onnx.py on CPU."
        ),
    )
    parser.add_argument(
        "--extra-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra option passed to onnxruntime-genai builder (repeatable).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
        help="Model load dtype for Optimum export backend.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-dir when it already exists.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip ONNX Runtime smoke test after export.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model_dir}")
    config_path = args.model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {args.model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    if model_type != "qwen3_5" and "Qwen3_5" not in "".join(architectures):
        raise ValueError(
            f"expected a Qwen3.5 checkpoint, got model_type={model_type!r} "
            f"architectures={architectures!r}"
        )

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}. Use --overwrite to refresh."
        )

    if args.cache_dir is None:
        args.cache_dir = PROJECT_ROOT / "models/.cache/onnxruntime-genai"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_text_chat_export_options(args)


def extra_option_keys(extra_options: list[str]) -> set[str]:
    keys: set[str] = set()
    for item in extra_options:
        key = item.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def ensure_text_chat_export_options(args: argparse.Namespace) -> None:
    if args.decoder_only:
        return
    if "exclude_embeds" in extra_option_keys(args.extra_option):
        return
    args.extra_option.append("exclude_embeds=false")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def require_transformers_for_qwen35() -> str:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Export requires transformers. Install with: pip install 'transformers>=5.3'"
        ) from exc

    version = parse_version(transformers.__version__)
    if version < MIN_TRANSFORMERS_VERSION:
        raise ImportError(
            f"Qwen3.5 export requires transformers>={'.'.join(map(str, MIN_TRANSFORMERS_VERSION))}, "
            f"but found {transformers.__version__}. "
            "Upgrade with: pip install 'transformers>=5.3'"
        )
    return transformers.__version__


def resolve_export_backend(backend_arg: str) -> str:
    if backend_arg == "auto":
        return "onnxruntime-genai"
    return backend_arg


def genai_builder_command(args: argparse.Namespace) -> list[str]:
    if args.genai_builder is not None:
        if not args.genai_builder.is_file():
            raise FileNotFoundError(f"genai builder script not found: {args.genai_builder}")
        return [sys.executable, str(args.genai_builder)]

    try:
        import onnxruntime_genai.models.builder  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnxruntime-genai export requires the onnxruntime-genai package. "
            "Install with: pip install --pre onnxruntime-genai onnx-ir\n"
            "Or provide --genai-builder /path/to/onnxruntime-genai/src/python/py/models/builder.py"
        ) from exc

    return [sys.executable, "-m", "onnxruntime_genai.models.builder"]


def needs_ortgenai_qwen35_text_arch_compat(model_dir: Path) -> bool:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    return (
        config.get("model_type") == "qwen3_5_text"
        and architectures[:1] == [ORTGENAI_QWEN35_TEXT_ARCH]
    )


def create_ortgenai_qwen35_text_compat_dir(
    model_dir: Path,
    *,
    parent_dir: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temp_dir = tempfile.TemporaryDirectory(
        prefix=".qwen35-ortgenai-",
        dir=str(parent_dir),
    )
    temp_path = Path(temp_dir.name)
    source_dir = model_dir.resolve()

    for source_path in source_dir.iterdir():
        if source_path.name == "config.json":
            continue
        target_path = temp_path / source_path.name
        relative_source = os.path.relpath(source_path, start=temp_path)
        os.symlink(relative_source, target_path)

    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    config["architectures"] = [ORTGENAI_QWEN35_BUILDER_ARCH]
    write_json(temp_path / "config.json", config)

    return temp_dir, temp_path


def export_with_onnxruntime_genai(args: argparse.Namespace) -> None:
    temp_model_dir: tempfile.TemporaryDirectory[str] | None = None
    model_dir = args.model_dir
    if needs_ortgenai_qwen35_text_arch_compat(args.model_dir):
        temp_model_dir, model_dir = create_ortgenai_qwen35_text_compat_dir(
            args.model_dir,
            parent_dir=args.output_dir.parent,
        )
        print(
            "using temporary Qwen3.5 text-only config compatibility directory for "
            "onnxruntime-genai builder"
        )

    command = genai_builder_command(args)
    command.extend(
        [
            "-i",
            str(model_dir),
            "-o",
            str(args.output_dir),
            "-p",
            args.precision,
            "-e",
            args.execution_provider,
            "-c",
            str(args.cache_dir),
        ]
    )
    if args.extra_option:
        command.extend(["--extra_options", *args.extra_option])

    print("running onnxruntime-genai builder:")
    print(" ", " ".join(command))
    try:
        subprocess.run(command, check=True)
    finally:
        if temp_model_dir is not None:
            temp_model_dir.cleanup()


def optimum_available() -> bool:
    try:
        from optimum.onnxruntime import ORTModelForCausalLM  # noqa: F401
    except ImportError:
        return False
    return True


def export_with_optimum(model_dir: Path, *, output_dir: Path, dtype_arg: str) -> None:
    if not optimum_available():
        raise ImportError(
            "Optimum export requires a compatible optimum install. "
            "Try: pip install \"optimum[onnxruntime]\""
        )

    import torch
    from optimum.onnxruntime import ORTModelForCausalLM

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_arg]

    print(
        "warning: Qwen3.5 hybrid decoder is usually unsupported by Optimum; "
        "prefer --export-backend onnxruntime-genai if this fails."
    )
    print(f"exporting with optimum from {model_dir}")
    ort_model = ORTModelForCausalLM.from_pretrained(
        str(model_dir),
        export=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    ort_model.save_pretrained(output_dir)


def list_onnx_files(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.glob("*.onnx"))


def validate_onnx_bundle(output_dir: Path) -> dict[str, Any]:
    import onnxruntime as ort

    onnx_files = list_onnx_files(output_dir)
    if not onnx_files:
        raise FileNotFoundError(f"no .onnx files found in {output_dir}")

    onnx_path = output_dir / onnx_files[0]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return {
        "runtime": "onnxruntime",
        "providers": session.get_providers(),
        "onnx_path": str(onnx_path),
        "onnx_files": onnx_files,
        "input_names": [item.name for item in session.get_inputs()],
        "output_names": [item.name for item in session.get_outputs()],
        "has_genai_config": (output_dir / "genai_config.json").is_file(),
    }


def load_model_metadata(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config") or {}
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "vocab_size": text_config.get("vocab_size", config.get("vocab_size")),
        "num_hidden_layers": text_config.get("num_hidden_layers", config.get("num_hidden_layers")),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_model_card(
    *,
    args: argparse.Namespace,
    export_backend: str,
    transformers_version: str,
    model_metadata: dict[str, Any],
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "model_version": args.model_version,
        "model_family": "qwen3_5",
        "status": "exported_onnx",
        "source_model_dir": str(args.model_dir),
        "export_backend": export_backend,
        "precision": args.precision,
        "execution_provider": args.execution_provider,
        "transformers_version": transformers_version,
        "exported_at_unix": int(time.time()),
        "metadata": model_metadata,
    }
    if validation is not None:
        card["export_validation"] = validation
    return card


def main() -> int:
    args = parse_args()
    validate_args(args)
    prepare_output_dir(args.output_dir, args.overwrite)

    transformers_version = require_transformers_for_qwen35()
    export_backend = resolve_export_backend(args.export_backend)
    model_metadata = load_model_metadata(args.model_dir)

    print(f"model_dir={args.model_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"transformers={transformers_version} export_backend={export_backend}")

    if export_backend == "onnxruntime-genai":
        export_with_onnxruntime_genai(args)
    else:
        export_with_optimum(
            args.model_dir,
            output_dir=args.output_dir,
            dtype_arg=args.dtype,
        )

    onnx_files = list_onnx_files(args.output_dir)
    if not onnx_files:
        raise RuntimeError("export finished but no .onnx files were found in output directory")

    validation: dict[str, Any] | None = None
    if not args.skip_validate:
        validation = validate_onnx_bundle(args.output_dir)
        print(
            "validated ONNX runtime: "
            f"files={validation['onnx_files']} "
            f"inputs={len(validation['input_names'])} "
            f"outputs={len(validation['output_names'])}"
        )

    model_card = build_model_card(
        args=args,
        export_backend=export_backend,
        transformers_version=transformers_version,
        model_metadata=model_metadata,
        validation=validation,
    )
    write_json(args.output_dir / "model_card.json", model_card)

    manifest = {
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "export_backend": export_backend,
        "precision": args.precision,
        "execution_provider": args.execution_provider,
        "onnx_files": onnx_files,
        "exported_at_unix": int(time.time()),
    }
    write_json(args.output_dir / "export_manifest.json", manifest)

    print(f"saved Qwen3.5 ONNX bundle to {args.output_dir}")
    print(f"onnx_files={onnx_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
