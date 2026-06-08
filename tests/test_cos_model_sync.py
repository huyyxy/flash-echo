from __future__ import annotations

import importlib.util
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cos_model_sync",
    PROJECT_ROOT / "tools" / "storage" / "cos_model_sync.py",
)
assert SPEC is not None
cos_model_sync = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cos_model_sync
SPEC.loader.exec_module(cos_model_sync)


class DummyResponse:
    status = 200

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


def test_upload_file_retries_transient_connection_error(monkeypatch, tmp_path) -> None:
    local_path = tmp_path / "model.safetensors"
    local_path.write_bytes(b"checkpoint")
    calls = []

    def fake_urlopen(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))
        return DummyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cos_model_sync.time, "sleep", lambda seconds: None)

    client = cos_model_sync.CosClient(
        bucket_url="https://weights-1305049745.cos.ap-shanghai.myqcloud.com",
        credentials=cos_model_sync.Credentials(secret_id="secret-id", secret_key="secret-key"),
        retries=1,
        retry_wait=0,
        timeout=30,
    )

    client.upload_file(local_path=local_path, key="flash-echo/model.safetensors")

    assert len(calls) == 2
    assert calls[0][1] == 30
    assert not isinstance(calls[0][0].data, bytes)


def test_upload_file_streams_data_and_reports_progress(monkeypatch, tmp_path, capsys) -> None:
    local_path = tmp_path / "model.onnx.data"
    local_path.write_bytes(b"x" * 1024 * 1024)
    uploaded = bytearray()

    def fake_urlopen(request, *, timeout):
        while chunk := request.data.read(128 * 1024):
            uploaded.extend(chunk)
        return DummyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = cos_model_sync.CosClient(
        bucket_url="https://weights-1305049745.cos.ap-shanghai.myqcloud.com",
        credentials=cos_model_sync.Credentials(secret_id="secret-id", secret_key="secret-key"),
        retries=0,
        retry_wait=0,
        timeout=600,
    )

    client.upload_file(local_path=local_path, key="flash-echo/model.onnx.data")

    assert bytes(uploaded) == local_path.read_bytes()
    assert "upload progress flash-echo/model.onnx.data: 1.0/1.0 MB" in capsys.readouterr().err
