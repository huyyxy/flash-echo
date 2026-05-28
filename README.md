# Flash Echo

Flash Echo is a low-latency filler words decision service for real-time voice interactions.

The service receives the current user `query` and `persona_tag`, predicts whether a filler prefix should be played, and returns a persona-specific prefix that downstream TTS and main LLM pipelines can use as the first spoken segment.

## Project Layout

```text
configs/                 Static service and template configuration
data/                    Raw, processed, and hard-case datasets
docs/                    Product requirements and implementation plan
models/                  Exported model packages and metadata
scripts/                 Data, template, training, and export utilities
src/filler_words/        Runtime package
tests/                   Unit and API tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn filler_words.app:create_app --factory --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/filler \
  -H "content-type: application/json" \
  -d '{"request_id":"req-001","query":"你怎么看 AI 对教育行业的影响？","persona_tag":"male_white_collar"}'
```

## Model Integration

The runtime currently ships with a conservative `HeuristicClassifier` so the service can run before the ONNX model package is available. Replace it with an ONNX-backed implementation behind `BaseClassifier` when `models/filler-cls-v1/` contains the exported model, tokenizer files, label mapping, and version metadata.
