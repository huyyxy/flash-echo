from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flash_echo.inference import minimind3_onnx
from flash_echo.inference.minimind3_onnx import MiniMind3ModelPool, MiniMind3OnnxGenerator


class FakeTensor:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)
        self.length = length

    def __getitem__(self, item: int | slice) -> FakeTensor:
        if isinstance(item, slice):
            start = item.start or 0
            stop = self.length if item.stop is None else item.stop
            return FakeTensor(max(stop - start, 0))
        return FakeTensor(self.length)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self,
        conversations: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return conversations[0]["content"]

    def __call__(self, prompt: str, **_: Any) -> dict[str, FakeTensor]:
        return {
            "input_ids": FakeTensor(3),
            "attention_mask": FakeTensor(3),
            "token_type_ids": FakeTensor(3),
        }

    def decode(self, generated_ids: FakeTensor, *, skip_special_tokens: bool) -> str:
        return "我想一下，"


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> list[FakeTensor]:
        self.generate_kwargs = kwargs
        return [FakeTensor(5)]


class SlowGenerator(MiniMind3OnnxGenerator):
    def _generate_sync(self, query: str):
        time.sleep(0.05)
        return super()._generate_sync(query)


class CountedGenerator:
    created = 0

    def __init__(self, bundle_dir: Path) -> None:
        time.sleep(0.01)
        type(self).created += 1
        self.bundle_dir = bundle_dir


def _write_inference_config(path: Path, *, timeout_ms: int = 500) -> None:
    (path / "inference_config.json").write_text(
        json.dumps(
            {
                "timeout_ms": timeout_ms,
                "prompt_template": {
                    "user_prefix": "",
                    "user_prompt_suffix": "",
                    "add_generation_prompt": True,
                },
                "generation": {
                    "max_seq_len": 64,
                    "max_new_tokens": 4,
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_generate_drops_token_type_ids_from_model_inputs(tmp_path: Path) -> None:
    _write_inference_config(tmp_path)
    fake_model = FakeModel()
    generator = MiniMind3OnnxGenerator(tmp_path)
    generator._model = fake_model
    generator._tokenizer = FakeTokenizer()

    result = generator._generate_sync("您好呀！")

    assert result.text == "我想一下，"
    assert fake_model.generate_kwargs is not None
    assert "input_ids" in fake_model.generate_kwargs
    assert "attention_mask" in fake_model.generate_kwargs
    assert "token_type_ids" not in fake_model.generate_kwargs


def test_timeout_replaces_executor(tmp_path: Path) -> None:
    _write_inference_config(tmp_path, timeout_ms=1)
    generator = SlowGenerator(tmp_path)
    generator._model = FakeModel()
    generator._tokenizer = FakeTokenizer()
    old_executor = generator._executor

    result = generator.generate("您好呀！")

    assert result.timed_out is True
    assert generator._executor is not old_executor


def test_model_pool_only_creates_one_session_under_concurrency(tmp_path: Path, monkeypatch) -> None:
    CountedGenerator.created = 0
    monkeypatch.setattr(minimind3_onnx, "MiniMind3OnnxGenerator", CountedGenerator)
    pool = MiniMind3ModelPool()

    with ThreadPoolExecutor(max_workers=8) as executor:
        sessions = list(executor.map(lambda _: pool.get(tmp_path), range(8)))

    assert CountedGenerator.created == 1
    assert len({id(session) for session in sessions}) == 1
