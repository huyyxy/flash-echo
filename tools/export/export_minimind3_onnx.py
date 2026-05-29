"""将 MiniMind3 SFT 微调 checkpoint 导出为 ONNX 部署包。

默认读取 ``tools/training/train_minimind3_sft.py`` 产出的
``models/checkpoints/minimind3-filler-<persona>-v1.0.0/best``，导出到
``models/deploy/minimind3-filler-<persona-hyphen>-v1.0.0/``。

部署包包含：

- ``model.onnx``：Optimum 导出的 text-generation-with-past 图（含 KV cache）
- tokenizer 文件（从 checkpoint 复制）
- ``inference_config.json``：短前缀生成默认参数与 Prompt 模板
- ``model_card.json`` / ``export_manifest.json``：版本与导出元数据

MiniMind3 基于 Qwen3 架构，推荐使用 Optimum 导出（``--export-backend auto``，默认）。
``torch.onnx.export`` 路径在 Qwen3 上通常不可用，仅保留作实验用途。

若 checkpoint 为 LoRA adapter（含 ``adapter_config.json``），脚本会在导出前自动
``merge_and_unload()``；必要时可通过 ``--base-model`` 指定基座模型。

使用示例::

    pip3 install -e ".[train,infer]"

    # 导出 SFT best checkpoint
    python3 tools/export/export_minimind3_onnx.py \\
      --persona male_white_collar

    # 指定 checkpoint 与输出目录
    python3 tools/export/export_minimind3_onnx.py \\
      --checkpoint models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/best \\
      --output-dir models/deploy/minimind3-filler-male-white-collar-v1.0.0 \\
      --overwrite

    # LoRA checkpoint：合并后再导出
    python3 tools/export/export_minimind3_onnx.py \\
      --checkpoint models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/best \\
      --base-model models/pretrained/jingyaogong-minimind-3
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSONA = "male_white_collar"
DEFAULT_PRETRAINED = PROJECT_ROOT / "models/pretrained/jingyaogong-minimind-3"
SFT_USER_PREFIX = "用户："
SFT_USER_PROMPT_SUFFIX = "请生成一句可续写的短垫话前缀："


class CausalLMLogitsWrapper(nn.Module):
    """导出单次前向：input_ids + attention_mask -> logits。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MiniMind3 SFT checkpoint to ONNX.")
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA,
        help="Persona slug for default checkpoint/output paths.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "SFT checkpoint directory (typically .../best). "
            "Default: models/checkpoints/minimind3-filler-<persona_underscore>-v1.0.0/best"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="ONNX bundle output directory. Default: models/deploy/minimind3-filler-<persona-hyphen>-v1.0.0",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Base MiniMind3 path or HF id for LoRA merge. "
            "Defaults to models/pretrained/jingyaogong-minimind-3 when present."
        ),
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Recorded in model_card.json. Default: minimind3-filler-<persona-hyphen>-v1.0.0",
    )
    parser.add_argument(
        "--export-backend",
        default="auto",
        choices=("auto", "optimum", "torch"),
        help="ONNX export backend. auto prefers optimum (recommended for Qwen3/MiniMind3).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version for torch export backend.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
        help="Model load dtype before export. float32 is most compatible.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Recorded in inference_config.json and used for export smoke input length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Default max generated tokens in inference_config.json.",
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
    parser.add_argument(
        "--skip-logits-check",
        action="store_true",
        help="Skip PyTorch vs ONNX logits numeric comparison (torch backend only).",
    )
    return parser.parse_args()


def default_base_model() -> str:
    if DEFAULT_PRETRAINED.exists():
        return str(DEFAULT_PRETRAINED)
    return "jingyaogong/minimind-3"


def resolve_paths(args: argparse.Namespace) -> None:
    checkpoint_version = f"minimind3-filler-{args.persona}-v1.0.0"
    deploy_version = args.model_version or f"minimind3-filler-{args.persona.replace('_', '-')}-v1.0.0"
    if args.model_version is None:
        args.model_version = deploy_version
    if args.checkpoint is None:
        args.checkpoint = PROJECT_ROOT / f"models/checkpoints/{checkpoint_version}/best"
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / f"models/deploy/{deploy_version}"


def validate_args(args: argparse.Namespace) -> None:
    resolve_paths(args)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {args.checkpoint}")
    if args.max_seq_len <= 0:
        raise ValueError("max-seq-len must be greater than 0")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be greater than 0")
    if args.opset < 14:
        raise ValueError("opset must be >= 14")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}. Use --overwrite to refresh."
        )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


def optimum_available() -> bool:
    try:
        import optimum.onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_export_backend(backend_arg: str) -> str:
    if backend_arg == "auto":
        if optimum_available():
            return "optimum"
        raise ImportError(
            "Optimum is required for MiniMind3/Qwen3 ONNX export. "
            "Install with: pip install \"optimum[onnxruntime]\""
        )
    if backend_arg == "optimum" and not optimum_available():
        raise ImportError(
            "Optimum export requires optimum. Install with: pip install \"optimum[onnxruntime]\""
        )
    return backend_arg


def is_lora_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


def load_training_model_card(checkpoint_dir: Path) -> dict[str, Any] | None:
    card_path = checkpoint_dir / "model_card.json"
    if not card_path.exists():
        parent_card = checkpoint_dir.parent / "model_card.json"
        if parent_card.exists():
            card_path = parent_card
        else:
            return None
    try:
        payload = json.loads(card_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def resolve_base_model_for_lora(
    *,
    checkpoint_dir: Path,
    base_model_arg: str | None,
    training_card: dict[str, Any] | None,
) -> str:
    if base_model_arg:
        return base_model_arg
    if training_card and isinstance(training_card.get("base_model"), str):
        return training_card["base_model"]
    return default_base_model()


def load_pytorch_model(
    checkpoint_dir: Path,
    *,
    base_model: str | None,
    dtype: torch.dtype,
) -> tuple[nn.Module, PreTrainedTokenizerBase, str]:
    training_card = load_training_model_card(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }

    if is_lora_checkpoint(checkpoint_dir):
        merge_base = resolve_base_model_for_lora(
            checkpoint_dir=checkpoint_dir,
            base_model_arg=base_model,
            training_card=training_card,
        )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "LoRA checkpoint export requires peft. Install with: pip install peft"
            ) from exc

        print(f"merging LoRA adapter from {checkpoint_dir} into base_model={merge_base}")
        base = AutoModelForCausalLM.from_pretrained(merge_base, **model_kwargs)
        model = PeftModel.from_pretrained(base, checkpoint_dir)
        model = model.merge_and_unload()
        source = f"lora:{checkpoint_dir}@base:{merge_base}"
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **model_kwargs)
        source = str(checkpoint_dir)

    model.eval()
    return model, tokenizer, source


@contextmanager
def merged_export_source(
    checkpoint_dir: Path,
    *,
    base_model: str | None,
    dtype: torch.dtype,
) -> Iterator[tuple[Path, PreTrainedTokenizerBase, str]]:
    if not is_lora_checkpoint(checkpoint_dir):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        yield checkpoint_dir, tokenizer, str(checkpoint_dir)
        return

    with tempfile.TemporaryDirectory(prefix="minimind3-merge-") as temp_dir:
        merged_dir = Path(temp_dir) / "merged"
        model, tokenizer, source = load_pytorch_model(
            checkpoint_dir,
            base_model=base_model,
            dtype=dtype,
        )
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        yield merged_dir, tokenizer, source


def build_sample_inputs(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    conversations = [
        {
            "role": "user",
            "content": f"{SFT_USER_PREFIX}今天天气怎么样？\n{SFT_USER_PROMPT_SUFFIX}",
        },
        {"role": "assistant", "content": "我先看一下，"},
    ]
    prompt = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


def export_with_torch_onnx(
    model: nn.Module,
    *,
    output_path: Path,
    sample_input_ids: torch.Tensor,
    sample_attention_mask: torch.Tensor,
    opset: int,
) -> None:
    wrapper = CausalLMLogitsWrapper(model)
    wrapper.eval()

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (sample_input_ids, sample_attention_mask),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def export_with_optimum(checkpoint_dir: Path, *, output_dir: Path) -> None:
    from optimum.onnxruntime import ORTModelForCausalLM

    print(f"exporting with optimum from {checkpoint_dir}")
    ort_model = ORTModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        export=True,
        trust_remote_code=True,
    )
    ort_model.save_pretrained(output_dir)


def copy_tokenizer_files(tokenizer: PreTrainedTokenizerBase, output_dir: Path) -> None:
    tokenizer.save_pretrained(output_dir)


def build_inference_config(args: argparse.Namespace, *, export_backend: str) -> dict[str, Any]:
    if export_backend == "optimum":
        onnx_meta = {
            "inputs": [
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values.*",
            ],
            "outputs": ["logits", "present.*"],
            "decoding": "kv_cache_autoregressive",
        }
    else:
        onnx_meta = {
            "inputs": ["input_ids", "attention_mask"],
            "outputs": ["logits"],
            "decoding": "greedy_last_token_loop",
        }

    return {
        "task": "filler_prefix_generation",
        "prompt_template": {
            "user_prefix": SFT_USER_PREFIX,
            "user_prompt_suffix": SFT_USER_PROMPT_SUFFIX,
            "chat_roles": ["user", "assistant"],
            "add_generation_prompt": True,
        },
        "generation": {
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": False,
            "repetition_penalty": 1.05,
        },
        "onnx": onnx_meta,
        "validation": {
            "max_chars": 48,
            "require_punctuation_endings": ["。", "，", "！", "？", "、"],
            "forbid_answer_patterns": [],
        },
        "timeout_ms": 500,
    }


def build_model_card(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    export_source: str,
    training_card: dict[str, Any] | None,
    export_backend: str,
    onnx_files: list[str],
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "model_version": args.model_version,
        "persona": args.persona,
        "model_family": "minimind3",
        "task": "filler_prefix_generation",
        "status": "exported_onnx",
        "source_checkpoint": str(checkpoint_dir),
        "export_source": export_source,
        "export_backend": export_backend,
        "exported_at_unix": int(time.time()),
        "onnx_files": onnx_files,
        "inference_config": "inference_config.json",
    }
    if training_card:
        for key in ("base_model", "training_mode", "metrics", "data", "parameters", "best_epoch"):
            if key in training_card:
                card[f"training_{key}"] = training_card[key]
    if validation is not None:
        card["export_validation"] = validation
    return card


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_torch_onnx_runtime(
    onnx_path: Path,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reference_logits: torch.Tensor | None,
    skip_logits_check: bool,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": input_ids.detach().cpu().numpy().astype("int64"),
        "attention_mask": attention_mask.detach().cpu().numpy().astype("int64"),
    }
    ort_outputs = session.run(None, ort_inputs)
    logits = ort_outputs[0]

    result: dict[str, Any] = {
        "runtime": "onnxruntime",
        "providers": session.get_providers(),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "input_names": [item.name for item in session.get_inputs()],
        "output_names": [item.name for item in session.get_outputs()],
    }

    if reference_logits is not None and not skip_logits_check:
        reference = reference_logits.detach().cpu().float().numpy()
        max_abs_diff = float(np.max(np.abs(reference - logits.astype(np.float32))))
        mean_abs_diff = float(np.mean(np.abs(reference - logits.astype(np.float32))))
        result["pytorch_max_abs_diff"] = max_abs_diff
        result["pytorch_mean_abs_diff"] = mean_abs_diff
        result["logits_close"] = max_abs_diff < 1e-3

    return result


def validate_optimum_onnx_bundle(output_dir: Path) -> dict[str, Any]:
    import onnxruntime as ort

    onnx_path = output_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"missing exported ONNX file: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]

    return {
        "runtime": "onnxruntime",
        "providers": session.get_providers(),
        "onnx_path": str(onnx_path),
        "input_names": input_names,
        "output_names": output_names,
        "has_kv_cache": any(name.startswith("past_key_values.") for name in input_names),
        "has_present_states": any(name.startswith("present.") for name in output_names),
    }


def list_onnx_files(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.glob("*.onnx"))


def main() -> int:
    try:
        import jinja2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MiniMind3 chat template requires jinja2. Install with: pip install jinja2"
        ) from exc

    args = parse_args()
    validate_args(args)
    prepare_output_dir(args.output_dir, args.overwrite)

    export_backend = resolve_export_backend(args.export_backend)
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cpu")
    print(f"checkpoint={args.checkpoint}")
    print(f"output_dir={args.output_dir}")
    print(f"dtype={dtype} export_backend={export_backend}")

    training_card = load_training_model_card(args.checkpoint)
    validation: dict[str, Any] | None = None

    with merged_export_source(
        args.checkpoint,
        base_model=args.base_model,
        dtype=dtype,
    ) as (export_source_dir, tokenizer, export_source):
        if export_backend == "optimum":
            export_with_optimum(export_source_dir, output_dir=args.output_dir)
            copy_tokenizer_files(tokenizer, args.output_dir)
            if not args.skip_validate:
                validation = validate_optimum_onnx_bundle(args.output_dir)
                print(
                    "validated ONNX runtime: "
                    f"inputs={len(validation['input_names'])} "
                    f"outputs={len(validation['output_names'])} "
                    f"kv_cache={validation['has_kv_cache']}"
                )
        else:
            model, _, _ = load_pytorch_model(
                export_source_dir,
                base_model=args.base_model,
                dtype=dtype,
            )
            model.to(device)
            sample_input_ids, sample_attention_mask = build_sample_inputs(
                tokenizer,
                max_seq_len=args.max_seq_len,
                device=device,
            )

            onnx_path = args.output_dir / "model.onnx"
            print(f"exporting torch ONNX to {onnx_path}")
            export_with_torch_onnx(
                model,
                output_path=onnx_path,
                sample_input_ids=sample_input_ids,
                sample_attention_mask=sample_attention_mask,
                opset=args.opset,
            )
            copy_tokenizer_files(tokenizer, args.output_dir)

            if not args.skip_validate:
                with torch.inference_mode():
                    reference_logits = model(
                        input_ids=sample_input_ids,
                        attention_mask=sample_attention_mask,
                        use_cache=False,
                    ).logits
                validation = validate_torch_onnx_runtime(
                    onnx_path,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    reference_logits=reference_logits,
                    skip_logits_check=args.skip_logits_check,
                )
                print(
                    "validated ONNX runtime: "
                    f"shape={validation['logits_shape']} "
                    f"max_abs_diff={validation.get('pytorch_max_abs_diff', 'n/a')}"
                )

    inference_config = build_inference_config(args, export_backend=export_backend)
    write_json(args.output_dir / "inference_config.json", inference_config)

    onnx_files = list_onnx_files(args.output_dir)
    if not onnx_files:
        raise RuntimeError("export finished but no .onnx files were found in output directory")

    model_card = build_model_card(
        args=args,
        checkpoint_dir=args.checkpoint,
        export_source=export_source,
        training_card=training_card,
        export_backend=export_backend,
        onnx_files=onnx_files,
        validation=validation,
    )
    write_json(args.output_dir / "model_card.json", model_card)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "export_backend": export_backend,
        "onnx_files": onnx_files,
        "exported_at_unix": int(time.time()),
    }
    write_json(args.output_dir / "export_manifest.json", manifest)

    print(f"saved MiniMind3 ONNX bundle to {args.output_dir}")
    print(f"onnx_files={onnx_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
