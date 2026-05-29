"""使用 MiniMind3 预训练权重进行短垫话前缀 SFT 微调。

默认读取 ``data/filler_prefix/<persona>/{train,valid}.jsonl``（由
``tools/training/prepare_sharegpt_filler_prefix.py`` 生成），基于本地
``models/pretrained/jingyaogong-minimind-3`` 做全参数 SFT；也可选 LoRA（需 ``peft``）。

训练样本格式（ShareGPT ``conversations``）::

    {
      "conversations": [
        {"role": "user", "content": "用户：{query}\\n请生成一句可续写的短垫话前缀："},
        {"role": "assistant", "content": "{filler_prefix}"}
      ]
    }

损失仅在 assistant 回复 token 上计算，逻辑与 MiniMind 官方 ``SFTDataset`` 一致。

使用示例::

    pip3 install -e ".[train]"
    python3 tools/training/download_minimind3_pretrained.py

    python3 tools/training/train_minimind3_sft.py \\
      --persona male_white_collar \\
      --train data/filler_prefix/male_white_collar/train.jsonl \\
      --valid data/filler_prefix/male_white_collar/valid.jsonl

    # LoRA（需 pip install peft）
    python3 tools/training/train_minimind3_sft.py --use-lora --lora-r 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED = PROJECT_ROOT / "models/pretrained/jingyaogong-minimind-3"
DEFAULT_PERSONA = "male_white_collar"
DEFAULT_FILLER_DATA_ROOT = PROJECT_ROOT / "data/filler_prefix"


@dataclass(frozen=True)
class ConversationSample:
    conversations: list[dict[str, str]]


class FillerPrefixSFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """MiniMind3 ShareGPT SFT：仅对 assistant 段计算 loss。"""

    def __init__(
        self,
        samples: list[ConversationSample],
        *,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if tokenizer.bos_token is None or tokenizer.eos_token is None:
            raise ValueError("tokenizer must define bos_token and eos_token")
        self.assistant_bos_ids = tokenizer(
            f"{tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        ).input_ids
        self.assistant_eos_ids = tokenizer(
            f"{tokenizer.eos_token}\n",
            add_special_tokens=False,
        ).input_ids
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")

    def __len__(self) -> int:
        return len(self.samples)

    def _render_chat(self, conversations: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _generate_labels(self, input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        bos = self.assistant_bos_ids
        eos = self.assistant_eos_ids
        index = 0
        while index < len(input_ids):
            if input_ids[index : index + len(bos)] == bos:
                start = index + len(bos)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(eos)] == eos:
                        break
                    end += 1
                upper = min(end + len(eos), self.max_seq_len)
                for position in range(start, upper):
                    labels[position] = input_ids[position]
                index = end + len(eos) if end < len(input_ids) else len(input_ids)
            else:
                index += 1
        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        conversations = self.samples[index].conversations
        prompt = self._render_chat(conversations)
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[: self.max_seq_len]
        labels = self._generate_labels(input_ids)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT fine-tune MiniMind3 for filler prefixes.")
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA,
        help="Persona slug for default paths and model_version naming.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Training JSONL. Default: data/filler_prefix/<persona>/train.jsonl",
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=None,
        help="Validation JSONL. Default: data/filler_prefix/<persona>/valid.jsonl",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Optional test JSONL for post-training eval.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "MiniMind3 pretrained path or HF id. Defaults to "
            "models/pretrained/jingyaogong-minimind-3 when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint root. Default: models/checkpoints/minimind3-filler-<persona>-v1.0.0",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Recorded in model_card.json. Default: minimind3-filler-<persona>-v1.0.0",
    )
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Model compute dtype. auto uses bf16 on CUDA when supported, else fp32.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Train LoRA adapters instead of full weights (requires peft).",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--save-last",
        action="store_true",
        help="Also save the final epoch under output-dir/last.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Debug: cap training samples.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Debug: cap validation samples.",
    )
    return parser.parse_args()


def default_base_model() -> str:
    if DEFAULT_PRETRAINED.exists():
        return str(DEFAULT_PRETRAINED)
    return "jingyaogong/minimind-3"


def resolve_paths(args: argparse.Namespace) -> None:
    persona_dir = DEFAULT_FILLER_DATA_ROOT / args.persona
    if args.train is None:
        args.train = persona_dir / "train.jsonl"
    if args.valid is None:
        args.valid = persona_dir / "valid.jsonl"
    if args.test is None:
        args.test = persona_dir / "test.jsonl"
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / f"models/checkpoints/minimind3-filler-{args.persona}-v1.0.0"
    if args.model_version is None:
        args.model_version = f"minimind3-filler-{args.persona}-v1.0.0"


def validate_args(args: argparse.Namespace) -> None:
    resolve_paths(args)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")
    if args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be greater than 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be greater than 0")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if args.max_grad_norm < 0:
        raise ValueError("max-grad-norm must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.max_seq_len <= 0:
        raise ValueError("max-seq-len must be greater than 0")
    for path in (args.train, args.valid):
        if not path.exists():
            raise FileNotFoundError(f"required data split does not exist: {path}")
    if args.use_lora and args.lora_r <= 0:
        raise ValueError("lora-r must be greater than 0 when --use-lora is set")


def load_conversations_jsonl(path: Path, *, max_samples: int | None = None) -> list[ConversationSample]:
    samples: list[ConversationSample] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: record must be a JSON object")
            conversations = record.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"{path}:{line_no}: conversations must be a list with >= 2 turns")
            parsed: list[dict[str, str]] = []
            for turn_index, turn in enumerate(conversations):
                if not isinstance(turn, dict):
                    raise ValueError(f"{path}:{line_no}: turn {turn_index} must be an object")
                role = turn.get("role")
                content = turn.get("content")
                if role not in ("user", "assistant", "system"):
                    raise ValueError(f"{path}:{line_no}: unsupported role: {role!r}")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{line_no}: content must be a non-empty string")
                parsed.append({"role": role, "content": content})
            if parsed[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{line_no}: last turn must be assistant")
            samples.append(ConversationSample(conversations=parsed))
            if max_samples is not None and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError(f"data split is empty: {path}")
    return samples


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    if device_arg == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but is not available")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # MPS 上 float16 训练易出现 NaN，默认使用 float32。
    return torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_sft_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(item[0].shape[0] for item in batch)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_mask: list[list[int]] = []

    for input_ids, labels in batch:
        pad_len = max_len - input_ids.shape[0]
        input_rows.append(input_ids.tolist() + [pad_token_id] * pad_len)
        label_rows.append(labels.tolist() + [-100] * pad_len)
        attention_mask.append([1] * input_ids.shape[0] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def make_loader(
    samples: list[ConversationSample],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = FillerPrefixSFTDataset(samples, tokenizer=tokenizer, max_seq_len=max_seq_len)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id is required")

    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_sft_batch, pad_token_id=pad_token_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def maybe_wrap_lora(
    model: torch.nn.Module,
    *,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> torch.nn.Module:
    if not use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires peft. Install with: pip install peft") from exc

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    return get_peft_model(model, config)


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        num = parameter.numel()
        total += num
        if parameter.requires_grad:
            trainable += num
    return trainable, total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = move_to_device(batch, device)
        labels = batch["labels"]
        outputs = model(**batch)
        batch_size = labels.shape[0]
        active = int((labels != -100).sum().item())
        total_loss += float(outputs.loss.detach().cpu()) * batch_size
        total_tokens += active

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "loss_per_token": total_loss / max(total_tokens, 1) if total_tokens else 0.0,
    }


def save_checkpoint(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_card(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    base_model: str,
    train_size: int,
    valid_size: int,
    test_size: int,
    trainable_params: int,
    total_params: int,
    best_valid_metrics: dict[str, float],
    best_epoch: int,
    test_metrics: dict[str, float] | None,
) -> None:
    card = {
        "model_version": args.model_version,
        "persona": args.persona,
        "base_model": base_model,
        "training_mode": "lora" if args.use_lora else "full_sft",
        "status": "trained",
        "best_checkpoint": str(args.output_dir / "best"),
        "best_epoch": best_epoch,
        "created_at_unix": int(time.time()),
        "data": {
            "train_path": str(args.train),
            "valid_path": str(args.valid),
            "test_path": str(args.test) if args.test.exists() else None,
            "train_size": train_size,
            "valid_size": valid_size,
            "test_size": test_size,
        },
        "parameters": {
            "trainable": trainable_params,
            "total": total_params,
        },
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "device"
        },
        "metrics": {
            "best_valid": best_valid_metrics,
            "test": test_metrics,
        },
    }
    (output_dir / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_grad_norm: float,
    accumulation_steps: int,
) -> float:
    model.train()
    total_loss_sum = 0.0
    total_loss_weight = 0.0
    window_loss_sum = 0.0
    window_loss_weight = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        batch_size = batch["labels"].shape[0]
        batch_loss = float(outputs.loss.detach().cpu()) * batch_size
        total_loss_sum += batch_loss
        total_loss_weight += batch_size
        window_loss_sum += batch_loss
        window_loss_weight += batch_size

        if step % accumulation_steps == 0 or step == len(loader):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 50 == 0 or step == len(loader):
            recent_loss = window_loss_sum / max(window_loss_weight, 1.0)
            print(
                f"  step {step}/{len(loader)} "
                f"loss={recent_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            window_loss_sum = 0.0
            window_loss_weight = 0.0

    return total_loss_sum / max(total_loss_weight, 1.0)


def main() -> int:
    try:
        import jinja2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MiniMind3 chat template requires jinja2. Install with: pip install jinja2"
        ) from exc

    args = parse_args()
    validate_args(args)
    base_model = args.base_model or default_base_model()

    seed_everything(args.seed)
    device = select_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    print(f"base_model={base_model}")
    print(f"device={device} dtype={model_dtype}")
    print(f"persona={args.persona} output_dir={args.output_dir}")

    train_samples = load_conversations_jsonl(args.train, max_samples=args.max_train_samples)
    valid_samples = load_conversations_jsonl(args.valid, max_samples=args.max_valid_samples)
    test_samples = (
        load_conversations_jsonl(args.test, max_samples=args.max_valid_samples)
        if args.test.exists()
        else []
    )
    print(
        f"loaded train={len(train_samples)} valid={len(valid_samples)} "
        f"test={len(test_samples)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=model_dtype,
    )
    model = maybe_wrap_lora(
        model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)
    trainable_params, total_params = trainable_parameter_count(model)
    print(f"trainable_params={trainable_params:,} total_params={total_params:,}")

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_samples,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = make_loader(
        valid_samples,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        make_loader(
            test_samples,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if test_samples
        else None
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = math.floor(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_score = float("inf")
    best_epoch = 0
    best_valid_metrics: dict[str, float] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_dir = args.output_dir / "best"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=args.max_grad_norm,
            accumulation_steps=args.gradient_accumulation_steps,
        )
        valid_metrics = evaluate(model, valid_loader, device=device)
        valid_metrics["train_loss"] = train_loss
        score = valid_metrics["loss"]
        print(
            "  valid "
            f"loss={valid_metrics['loss']:.4f} "
            f"loss_per_token={valid_metrics['loss_per_token']:.4f}"
        )
        if math.isfinite(score) and score < best_score:
            best_score = score
            best_epoch = epoch
            best_valid_metrics = valid_metrics
            best_model_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            print(f"  updated best checkpoint candidate for {best_dir}")

    if args.save_last:
        save_checkpoint(args.output_dir / "last", model=model, tokenizer=tokenizer)

    if best_valid_metrics is None or best_model_state is None:
        raise RuntimeError(
            "training finished without a finite validation loss; "
            "try --dtype float32 or a smaller learning-rate"
        )

    model.load_state_dict(best_model_state)
    model.to(device)
    save_checkpoint(best_dir, model=model, tokenizer=tokenizer)

    test_metrics = evaluate(model, test_loader, device=device) if test_loader is not None else None
    if test_metrics is not None:
        print(
            "test "
            f"loss={test_metrics['loss']:.4f} "
            f"loss_per_token={test_metrics['loss_per_token']:.4f}"
        )

    for metadata_dir in (best_dir, args.output_dir):
        metadata_dir.mkdir(parents=True, exist_ok=True)
        save_model_card(
            metadata_dir,
            args=args,
            base_model=base_model,
            train_size=len(train_samples),
            valid_size=len(valid_samples),
            test_size=len(test_samples),
            trainable_params=trainable_params,
            total_params=total_params,
            best_valid_metrics=best_valid_metrics,
            best_epoch=best_epoch,
            test_metrics=test_metrics,
        )

    print(f"done. best checkpoint: {best_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
