from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filler_words.inference.minimind3_onnx import MiniMind3OnnxGenerator


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


def test_generate_drops_token_type_ids_from_model_inputs(tmp_path: Path) -> None:
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {
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
