FROM python:3.11-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/models/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/models/.cache/huggingface \
    ORTGENAI_CACHE=/workspace/models/.cache/onnxruntime-genai

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging

# Install Linux x86_64 CPU PyTorch from the official CPU wheel index. This keeps
# the export stack independent of Intel macOS PyTorch wheel availability.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

# Qwen3.5 support lives in newer/pre-release HF and ORT GenAI stacks. Do not use
# this project's current optional extras here because they intentionally cap
# transformers below 5.x for the app/runtime environment.
RUN python -m pip install --pre \
    "transformers>=5.3" \
    "onnxruntime>=1.18" \
    "onnxruntime-genai" \
    "onnx-ir" \
    "onnx" \
    "huggingface-hub" \
    "safetensors" \
    "tokenizers" \
    "accelerate"

CMD ["bash"]
