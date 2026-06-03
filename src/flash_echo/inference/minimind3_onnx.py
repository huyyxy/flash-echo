from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash_echo.inference.validator import OutputValidator, load_inference_config


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    timed_out: bool = False


class MiniMind3OnnxGenerator:
    """基于 Optimum 导出的 MiniMind3 SFT ONNX 包进行短前缀自回归生成。"""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        self.inference_config = load_inference_config(bundle_dir)
        self.validator = OutputValidator.from_inference_config(self.inference_config)
        self.model_version = self._load_model_version(bundle_dir)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self._executor = self._create_executor()

    @staticmethod
    def _create_executor() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="minimind3-onnx")

    @staticmethod
    def _load_model_version(bundle_dir: Path) -> str:
        card_path = bundle_dir / "model_card.json"
        if card_path.exists():
            payload = json.loads(card_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("model_version"), str):
                return payload["model_version"]
        return bundle_dir.name

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            try:
                from optimum.onnxruntime import ORTModelForCausalLM
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "MiniMind3 ONNX inference requires optimum and transformers. "
                    'Install with: pip install -e ".[infer]"'
                ) from exc

            self._model = ORTModelForCausalLM.from_pretrained(
                str(self.bundle_dir),
                provider="CPUExecutionProvider",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.bundle_dir),
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    def build_prompt(self, query: str) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None

        template = self.inference_config["prompt_template"]
        user_content = (
            f"{template['user_prefix']}{query}\n{template['user_prompt_suffix']}"
        )
        conversations = [{"role": "user", "content": user_content}]
        return self._tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=bool(template.get("add_generation_prompt", True)),
        )

    def generate(self, query: str) -> GenerationResult:
        timeout_ms = int(self.inference_config.get("timeout_ms", 500))
        timeout_seconds = max(timeout_ms, 1) / 1000.0
        with self._executor_lock:
            future = self._executor.submit(self._generate_sync, query)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            future.cancel()
            self._replace_executor()
            return GenerationResult(text="", prompt_tokens=0, completion_tokens=0, timed_out=True)

    def _replace_executor(self) -> None:
        with self._executor_lock:
            old_executor = self._executor
            self._executor = self._create_executor()
        old_executor.shutdown(wait=False, cancel_futures=True)

    def _generate_sync(self, query: str) -> GenerationResult:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        generation_cfg = self.inference_config["generation"]
        prompt = self.build_prompt(query)
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(generation_cfg["max_seq_len"]),
        )
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        model_inputs = {
            key: encoded[key]
            for key in ("input_ids", "attention_mask")
            if key in encoded
        }

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(generation_cfg["max_new_tokens"]),
            "do_sample": bool(generation_cfg.get("do_sample", False)),
            "repetition_penalty": float(generation_cfg.get("repetition_penalty", 1.0)),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
            generate_kwargs["top_p"] = float(generation_cfg.get("top_p", 0.9))

        output_ids = self._model.generate(**model_inputs, **generate_kwargs)
        generated_ids = output_ids[0][prompt_tokens:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = int(generated_ids.shape[-1])
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class MiniMind3ModelPool:
    """按 Persona 懒加载 ONNX 生成器，避免启动时加载全部模型。"""

    def __init__(self) -> None:
        self._sessions: dict[str, MiniMind3OnnxGenerator] = {}
        self._lock = threading.Lock()

    def get(self, bundle_dir: Path) -> MiniMind3OnnxGenerator:
        key = str(bundle_dir.resolve())
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = MiniMind3OnnxGenerator(bundle_dir)
                self._sessions[key] = session
            return session
