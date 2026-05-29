# Flash Echo

Flash Echo 是一个用于实时语音交互的低延迟语气词决策服务。

该服务接收当前用户的 `query` 和 `persona_tag`，预测是否需要播放语气词前缀，并返回一个与 persona 对应的前缀，供下游 TTS 与主 LLM 流水线作为首段语音内容使用。

## 项目结构

```text
configs/                 运行时配置（路由、静态规则、fallback）
data/                    原始语料与 SFT 数据集
docs/                    产品需求、训练与推理文档
models/deploy/           ONNX 部署包（推理服务唯一消费的模型制品）
src/filler_words/        推理服务运行时
tools/training/          MiniMind 数据生成、下载、SFT/LoRA 训练
tools/export/            checkpoint -> ONNX deploy bundle 导出
tests/                   单元测试与 API 测试
```

训练与推理使用不同依赖环境，详见下方「环境安装」。

## 环境安装

### 推理服务（线上 / 本地开发）

```bash
python -m venv .venv-infer
source .venv-infer/bin/activate
pip install -e ".[dev,infer]"
```

仅需 FastAPI 与 ONNX Runtime 相关依赖，**不需要** PyTorch。

### 训练环境

```bash
python -m venv .venv-train
source .venv-train/bin/activate
pip install -e ".[train]"
# LoRA 微调额外需要：pip install peft
# ModelScope 下载额外需要：pip install modelscope
```

### ONNX 导出

导出脚本同时需要训练与推理依赖：

```bash
pip install -e ".[train,infer]"
```

兼容旧命令：`pip install -e ".[ml]"` 等价于安装训练 + 推理全部 ML 依赖。

## 快速开始（推理服务）

```bash
source .venv-infer/bin/activate
uvicorn filler_words.app:create_app --factory --reload
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "filler-reply-minimind3",
    "messages": [{"role": "user", "content": "你怎么看 AI 对教育行业的影响？"}],
    "metadata": {"persona_tag": "male_white_collar", "request_id": "req-001"}
  }'
```

## 模型集成

推理服务通过 `configs/model_router.v1.json` 将 `persona_tag` 映射到 `models/deploy/<model_version>/` 下的 ONNX bundle。若对应 deploy 包不存在，服务会自动回退到静态规则或 fallback 前缀。

完整训练、导出与部署流程见：

- [docs/training.md](docs/training.md) — MiniMind 数据生成、预训练下载、SFT/LoRA
- [docs/inference.md](docs/inference.md) — ONNX 导出、deploy bundle、服务配置

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FILLER_CONFIG_DIR` | `configs/` | 运行时配置文件目录 |
| `FILLER_DEPLOY_ROOT` | `models/deploy/` | ONNX 部署包根目录 |
| `FILLER_HOST` | `0.0.0.0` | 服务监听地址 |
| `FILLER_PORT` | `8000` | 服务端口 |
