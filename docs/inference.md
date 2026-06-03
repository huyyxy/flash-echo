# ONNX 推理与部署

本文档描述 Flash Echo 推理环境的配置与 ONNX 模型部署流程。推理服务**只消费** `models/deploy/` 下的 ONNX bundle，不直接读取训练 checkpoint。

## 环境准备

### 仅运行推理服务

```bash
python -m venv .venv-infer
source .venv-infer/bin/activate
pip install -e ".[dev,infer]"
```

### 导出 ONNX（需要训练 + 推理依赖）

```bash
pip install -e ".[train,infer]"
```

## 导出 ONNX 部署包

SFT 完成后，将 best checkpoint 导出为线上推理用的 ONNX 包：

```bash
python3 tools/export/export_minimind3_onnx.py \
  --persona male_white_collar
```

脚本默认：

- 读取 checkpoint：`models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/best`
- 输出 deploy 包：`models/deploy/minimind3-filler-male-white-collar-v1.0.0/`

也可显式指定路径：

```bash
python3 tools/export/export_minimind3_onnx.py \
  --persona male_white_collar \
  --checkpoint models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/best \
  --model-version minimind3-filler-male-white-collar-v1.0.0 \
  --output-dir models/deploy/minimind3-filler-male-white-collar-v1.0.0
```

LoRA checkpoint 会在导出前自动合并；必要时加上 `--base-model models/pretrained/jingyaogong-minimind-3`。

## Deploy Bundle 结构

```text
models/deploy/<model_version>/
  model.onnx                  Optimum 导出的 text-generation 图
  tokenizer files             从 checkpoint 复制的 tokenizer
  inference_config.json       生成参数与 Prompt 模板
  model_card.json             版本元数据
  export_manifest.json        导出记录
```

## 运行时配置

推理服务启动时读取 `configs/` 下的配置文件：

| 文件 | 说明 |
|------|------|
| `model_router.v1.json` | `persona_tag` -> `model_version` 路由 |
| `static_rules.v1.json` | 静态规则（问候、感谢等） |
| `fallback_prefixes.v1.json` | 模型不可用时的兜底前缀 |

`model_router.v1.json` 示例：

```json
{
  "version": "model-router-v1",
  "routes": {
    "male_white_collar": {
      "model_version": "minimind3-filler-male-white-collar-v1.0.0",
      "enabled": true
    }
  }
}
```

路由解析时会检查 `models/deploy/<model_version>/model.onnx` 是否存在；不存在则自动 fallback。

可通过环境变量覆盖路径：

```bash
export FILLER_CONFIG_DIR=configs
export FILLER_DEPLOY_ROOT=models/deploy
uvicorn flash_echo.app:create_app --factory --reload
```

## 启动推理服务

```bash
source .venv-infer/bin/activate
uvicorn flash_echo.app:create_app --factory --host 0.0.0.0 --port 8000
```

或使用 CLI 入口：

```bash
flash-echo
```

## Persona 命名约定

| 场景 | 格式 | 示例 |
|------|------|------|
| 数据 / checkpoint | 下划线 | `male_white_collar` |
| deploy / model_version | 连字符 | `minimind3-filler-male-white-collar-v1.0.0` |

新增 persona 时：更新 `configs/model_router.v1.json`，导出对应 deploy 包，无需修改服务代码。
