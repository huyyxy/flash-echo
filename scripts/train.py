"""Train the filler type classifier.

默认读取 ``data/processed/{train,valid,test}.jsonl``，基于 ``hfl/rbt3`` 或本地
预下载模型微调一个七分类模型，并输出 Hugging Face checkpoint、``labels.json``
和 ``model_card.json``。

使用示例::

    pip3 install -e ".[ml]"
    python3 scripts/download_pretrained_model.py
    python3 scripts/train.py --base-model models/pretrained/hfl-rbt3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from filler_words.core.labels import FillerType, trigger_for_type


LABELS = [label.value for label in FillerType]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Sample:
    query: str
    label_id: int
    filler_type: str
    trigger: int
    source: str | None = None
    label_method: str | None = None


class FillerDataset(Dataset[Sample]):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the flash-echo filler classifier.")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data/processed/train.jsonl")
    parser.add_argument("--valid", type=Path, default=PROJECT_ROOT / "data/processed/valid.jsonl")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/processed/test.jsonl")
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Base model name or local path. Defaults to models/pretrained/hfl-rbt3 "
            "when present, otherwise hfl/rbt3."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models/checkpoints/filler-cls-v1",
    )
    parser.add_argument("--model-version", default="filler-cls-v1")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training device. auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights in the cross entropy loss.",
    )
    parser.add_argument(
        "--save-last",
        action="store_true",
        help="Also save the final epoch checkpoint under output-dir/last.",
    )
    return parser.parse_args()


def default_base_model() -> str:
    local_model = PROJECT_ROOT / "models/pretrained/hfl-rbt3"
    return str(local_model) if local_model.exists() else "hfl/rbt3"


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than 0")
    if args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be greater than 0")
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
    if args.max_length <= 0:
        raise ValueError("max-length must be greater than 0")
    for path in (args.train, args.valid):
        if not path.exists():
            raise FileNotFoundError(f"required data split does not exist: {path}")


def normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def sample_from_record(record: dict[str, Any], *, path: Path, line_no: int) -> Sample:
    query = record.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{path}:{line_no}: query must be a non-empty string")

    filler_type = record.get("filler_type")
    if filler_type not in LABEL_TO_ID:
        raise ValueError(f"{path}:{line_no}: unknown filler_type: {filler_type!r}")

    expected_trigger = trigger_for_type(FillerType(filler_type))
    trigger = record.get("trigger", expected_trigger)
    if not isinstance(trigger, int) or isinstance(trigger, bool) or trigger not in (0, 1):
        raise ValueError(f"{path}:{line_no}: trigger must be integer 0 or 1")
    if trigger != expected_trigger:
        raise ValueError(
            f"{path}:{line_no}: inconsistent trigger={trigger} for filler_type={filler_type}"
        )

    source = record.get("source")
    label_method = record.get("label_method")
    return Sample(
        query=normalize_query(query),
        label_id=LABEL_TO_ID[filler_type],
        filler_type=filler_type,
        trigger=trigger,
        source=source if isinstance(source, str) else None,
        label_method=label_method if isinstance(label_method, str) else None,
    )


def load_jsonl(path: Path) -> list[Sample]:
    samples: list[Sample] = []
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
            samples.append(sample_from_record(record, path=path, line_no=line_no))
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


def collate_batch(
    samples: list[Sample],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [sample.query for sample in samples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([sample.label_id for sample in samples], dtype=torch.long)
    return encoded


def make_loader(
    samples: list[Sample],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(
        FillerDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_batch, tokenizer=tokenizer, max_length=max_length),
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def class_weights(samples: Iterable[Sample], *, device: torch.device) -> torch.Tensor:
    counts = Counter(sample.label_id for sample in samples)
    total = sum(counts.values())
    missing_labels = [ID_TO_LABEL[label_id] for label_id in range(len(LABELS)) if counts[label_id] == 0]
    if missing_labels:
        warnings.warn(
            "training split is missing labels; the model cannot learn these classes: "
            + ", ".join(missing_labels),
            stacklevel=2,
        )
    weights = []
    for label_id in range(len(LABELS)):
        count = counts[label_id]
        weights.append(total / (len(LABELS) * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float, device=device)


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot compute metrics on an empty dataset")

    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for true_id, pred_id in zip(y_true, y_pred, strict=True):
        confusion[true_id][pred_id] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    supported_f1_values: list[float] = []
    all_f1_values: list[float] = []
    for label_id, label in ID_TO_LABEL.items():
        tp = confusion[label_id][label_id]
        fp = sum(confusion[row][label_id] for row in range(len(LABELS)) if row != label_id)
        fn = sum(confusion[label_id][col] for col in range(len(LABELS)) if col != label_id)
        scores = precision_recall_f1(tp, fp, fn)
        support = sum(confusion[label_id])
        per_class[label] = {**scores, "support": support}
        all_f1_values.append(scores["f1"])
        if support > 0:
            supported_f1_values.append(scores["f1"])

    none_id = LABEL_TO_ID[FillerType.NONE.value]
    trigger_tp = sum(
        1 for true_id, pred_id in zip(y_true, y_pred, strict=True) if true_id != none_id and pred_id != none_id
    )
    trigger_fp = sum(
        1 for true_id, pred_id in zip(y_true, y_pred, strict=True) if true_id == none_id and pred_id != none_id
    )
    trigger_fn = sum(
        1 for true_id, pred_id in zip(y_true, y_pred, strict=True) if true_id != none_id and pred_id == none_id
    )

    return {
        "accuracy": sum(1 for true_id, pred_id in zip(y_true, y_pred, strict=True) if true_id == pred_id)
        / len(y_true),
        "type_macro_f1": sum(supported_f1_values) / len(supported_f1_values),
        "type_macro_f1_all_labels": sum(all_f1_values) / len(all_f1_values),
        "trigger": precision_recall_f1(trigger_tp, trigger_fp, trigger_fn),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "labels": LABELS,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        batch_size = batch["labels"].shape[0]
        total_loss += float(outputs.loss.detach().cpu()) * batch_size
        total_examples += batch_size
        predictions = outputs.logits.argmax(dim=-1)
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_examples
    return metrics


def save_labels(output_dir: Path) -> None:
    payload = {
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
    }
    (output_dir / "labels.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_model_card(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    base_model: str,
    train_samples: list[Sample],
    valid_samples: list[Sample],
    test_samples: list[Sample] | None,
    best_valid_metrics: dict[str, Any],
    best_epoch: int,
    test_metrics: dict[str, Any] | None,
) -> None:
    card = {
        "model_version": args.model_version,
        "base_model": base_model,
        "status": "trained",
        "best_checkpoint": str(args.output_dir / "best"),
        "best_epoch": best_epoch,
        "created_at_unix": int(time.time()),
        "labels": LABELS,
        "data": {
            "train_path": str(args.train),
            "valid_path": str(args.valid),
            "test_path": str(args.test) if args.test.exists() else None,
            "train_size": len(train_samples),
            "valid_size": len(valid_samples),
            "test_size": len(test_samples) if test_samples is not None else 0,
            "train_label_counts": dict(sorted(Counter(sample.filler_type for sample in train_samples).items())),
            "valid_label_counts": dict(sorted(Counter(sample.filler_type for sample in valid_samples).items())),
            "test_label_counts": (
                dict(sorted(Counter(sample.filler_type for sample in test_samples).items()))
                if test_samples is not None
                else {}
            ),
        },
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"device"}
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


def save_checkpoint(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_labels(output_dir)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    loss_fn: torch.nn.CrossEntropyLoss | None,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss_sum = 0.0
    total_loss_weight = 0.0
    window_loss_sum = 0.0
    window_loss_weight = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        labels = batch["labels"]
        optimizer.zero_grad(set_to_none=True)

        if loss_fn is None:
            outputs = model(**batch)
            loss = outputs.loss
        else:
            model_inputs = {key: value for key, value in batch.items() if key != "labels"}
            outputs = model(**model_inputs)
            loss = loss_fn(outputs.logits, labels)

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_size = labels.shape[0]
        if loss_fn is not None and loss_fn.weight is not None:
            batch_weight = float(loss_fn.weight[labels].detach().sum().cpu())
        else:
            batch_weight = float(batch_size)
        batch_loss_sum = float(loss.detach().cpu()) * batch_weight
        total_loss_sum += batch_loss_sum
        total_loss_weight += batch_weight
        window_loss_sum += batch_loss_sum
        window_loss_weight += batch_weight

        if step == 1 or step % 50 == 0 or step == len(loader):
            recent_loss = window_loss_sum / max(window_loss_weight, 1.0)
            print(
                f"  step {step}/{len(loader)} "
                f"loss={recent_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            window_loss_sum = 0.0
            window_loss_weight = 0.0

    return total_loss_sum / total_loss_weight


def main() -> int:
    args = parse_args()
    validate_args(args)
    base_model = args.base_model or default_base_model()

    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"base_model={base_model}")
    print(f"device={device}")

    train_samples = load_jsonl(args.train)
    valid_samples = load_jsonl(args.valid)
    test_samples = load_jsonl(args.test) if args.test.exists() else None
    print(f"loaded train={len(train_samples)} valid={len(valid_samples)} test={len(test_samples or [])}")
    print(f"train_label_counts={dict(sorted(Counter(sample.filler_type for sample in train_samples).items()))}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        train_samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = make_loader(
        valid_samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        make_loader(
            test_samples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if test_samples is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = math.floor(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = None
    if not args.no_class_weights:
        weights = class_weights(train_samples, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        print(f"class_weights={[round(float(value), 4) for value in weights.detach().cpu()]}")

    best_score = -1.0
    best_epoch = 0
    best_valid_metrics: dict[str, Any] | None = None
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
            loss_fn=loss_fn,
            max_grad_norm=args.max_grad_norm,
        )
        valid_metrics = evaluate(model, valid_loader, device=device)
        valid_metrics["train_loss"] = train_loss
        score = valid_metrics["type_macro_f1"]
        print(
            "  valid "
            f"loss={valid_metrics['loss']:.4f} "
            f"accuracy={valid_metrics['accuracy']:.4f} "
            f"type_macro_f1={valid_metrics['type_macro_f1']:.4f} "
            f"trigger_precision={valid_metrics['trigger']['precision']:.4f} "
            f"trigger_recall={valid_metrics['trigger']['recall']:.4f}"
        )
        if score > best_score:
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
        raise RuntimeError("training finished without validation metrics")

    model.load_state_dict(best_model_state)
    model.to(device)
    save_checkpoint(best_dir, model=model, tokenizer=tokenizer)
    test_metrics = evaluate(model, test_loader, device=device) if test_loader is not None else None
    if test_metrics is not None:
        print(
            "test "
            f"loss={test_metrics['loss']:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f} "
            f"type_macro_f1={test_metrics['type_macro_f1']:.4f} "
            f"trigger_precision={test_metrics['trigger']['precision']:.4f} "
            f"trigger_recall={test_metrics['trigger']['recall']:.4f}"
        )

    for metadata_dir in (best_dir, args.output_dir):
        metadata_dir.mkdir(parents=True, exist_ok=True)
        save_model_card(
            metadata_dir,
            args=args,
            base_model=base_model,
            train_samples=train_samples,
            valid_samples=valid_samples,
            test_samples=test_samples,
            best_valid_metrics=best_valid_metrics,
            best_epoch=best_epoch,
            test_metrics=test_metrics,
        )
    save_labels(args.output_dir)
    print(f"done. best checkpoint: {best_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
