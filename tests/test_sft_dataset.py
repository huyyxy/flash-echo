from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "tools/training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from filler_prefix_sft import ConversationSample, FillerPrefixSFTDataset  # noqa: E402


class Encoded:
    def __init__(self, input_ids: list[int], offset_mapping: list[tuple[int, int]]) -> None:
        self.input_ids = input_ids
        self.offset_mapping = offset_mapping


class FakeChatTokenizer:
    pad_token_id = 0
    eos_token = "<|im_end|>"
    sep_token = None
    special_tokens_map = {
        "eos_token": "<|im_end|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }

    def apply_chat_template(
        self,
        conversations: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        rendered = ""
        for message in conversations:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        return rendered

    def __call__(
        self,
        prompt: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Encoded:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        tokens = [
            "<|im_start|>",
            "user",
            "\n",
            "您好！",
            "<|im_end|>",
            "\n",
            "<|im_start|>",
            "assistant",
            "\n",
            "让我想想,",
            "<|im_end|>",
            "\n",
        ]

        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for index, token in enumerate(tokens, start=1):
            start = prompt.find(token, cursor)
            assert start >= 0
            end = start + len(token)
            input_ids.append(index)
            offsets.append((start, end))
            cursor = end
        return Encoded(input_ids=input_ids, offset_mapping=offsets)


def test_sft_labels_include_assistant_end_token() -> None:
    dataset = FillerPrefixSFTDataset(
        [
            ConversationSample(
                conversations=[
                    {"role": "user", "content": "您好！"},
                    {"role": "assistant", "content": "让我想想,"},
                ]
            )
        ],
        tokenizer=FakeChatTokenizer(),  # type: ignore[arg-type]
        max_seq_len=128,
    )

    input_ids, labels = dataset[0]

    assert labels.tolist() == [
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        10,
        11,
        -100,
    ]
    assert input_ids.tolist()[10] == 11
