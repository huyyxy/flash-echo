from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "tools/inference/serve_qwen3_5_onnx_openai.py"


def load_server_module() -> Any:
    module_name = "serve_qwen3_5_onnx_openai"
    spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen_chat_completions_accepts_any_model_name(monkeypatch) -> None:
    server = load_server_module()
    settings = server.ServerSettings(
        model_dir=PROJECT_ROOT / "models/deploy/fake-qwen",
        model_name="qwen3.5-0.8b",
        host="127.0.0.1",
        port=8000,
        max_context_tokens=4096,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        execution_provider="cpu",
        enable_thinking=False,
    )
    runtime = SimpleNamespace(lock=threading.Lock())

    monkeypatch.setattr(
        server,
        "generate_completion",
        lambda *_args, **_kwargs: server.qwen_chat.GenerationStats(
            text="好的，",
            prompt_tokens=3,
            completion_tokens=2,
        ),
    )

    client = TestClient(server.create_app(settings, runtime))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "anything-client-sends",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "qwen3.5-0.8b"
    assert payload["choices"][0]["message"]["content"] == "好的，"
