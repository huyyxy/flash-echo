import argparse
import json

import pytest

from scripts.prepare_sharegpt_jsonl import (
    iter_user_queries,
    validate_llm_labels,
    validate_runtime_args,
)


def test_iter_user_queries_accepts_openai_and_sharegpt_turns(tmp_path) -> None:
    input_path = tmp_path / "raw.jsonl"
    records = [
        {
            "conversations": [
                {"role": "user", "content": "openai-style"},
                {"role": "assistant", "content": "skip me"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "sharegpt-style"},
                {"from": "gpt", "value": "skip me too"},
            ]
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    assert list(iter_user_queries(input_path)) == [
        (1, "openai-style"),
        (2, "sharegpt-style"),
    ]


def test_validate_llm_labels_rejects_duplicate_ids() -> None:
    response = {
        "labels": [
            {"id": 0, "trigger": 0, "filler_type": "NONE"},
            {"id": 0, "trigger": 0, "filler_type": "NONE"},
            {"id": 1, "trigger": 1, "filler_type": "ACKNOWLEDGE"},
        ]
    }

    with pytest.raises(ValueError, match="duplicate label id"):
        validate_llm_labels(json.dumps(response), [(1, "a"), (2, "b")])


def test_validate_llm_labels_rejects_bool_trigger() -> None:
    response = {"labels": [{"id": 0, "trigger": False, "filler_type": "NONE"}]}

    with pytest.raises(ValueError, match="trigger"):
        validate_llm_labels(json.dumps(response), [(1, "a")])


def test_validate_runtime_args_rejects_values_before_outputs_opened() -> None:
    args = argparse.Namespace(
        batch_size=0,
        retries=1,
        retry_wait=0,
        timeout=1,
        min_query_chars=1,
        max_records=0,
    )

    with pytest.raises(ValueError, match="batch-size"):
        validate_runtime_args(args)
