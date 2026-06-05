# Flash Echo

Flash Echo 是一个用于实时语音交互的低延迟语气词决策服务。

该服务接收当前用户的 `query` 和 `persona_tag`，预测是否需要播放语气词前缀，并返回一个与 persona 对应的前缀，供下游 TTS 与主 LLM 流水线作为首段语音内容使用。

## 项目结构

```text
configs/                 运行时配置与模型流水线配置
configs/pipelines/       MiniMind3 / Qwen3.5-0.8B 流水线定义
configs/runtimes/        local / Docker CPU / Docker CUDA 等执行环境定义
data/                    原始语料与 SFT 数据集
docker/                  训练、导出、推理镜像的 Dockerfile
docs/                    产品需求、训练、推理与流水线文档
models/deploy/           ONNX / ORT GenAI 部署包（推理服务唯一消费的模型制品）
src/flash_echo_pipeline/ 模型数据、训练、导出、推理流水线 CLI
src/flash_echo/          推理服务运行时
tools/training/          数据生成、预训练下载、SFT/LoRA 训练 CLI
tools/export/            checkpoint -> ONNX deploy bundle 导出 CLI
tools/inference/         本地推理调试与 OpenAI 兼容服务 CLI
tests/                   单元测试与 API 测试
```

训练、导出、推理可以使用不同依赖环境。推荐通过 `flash-echo-pipeline` 统一编排步骤，并按需要选择本地或 Docker runtime。`tools/` 下脚本作为稳定 CLI 入口保留，后续共享逻辑会逐步沉到 `src/flash_echo_pipeline/steps/`。

## 环境安装

项目要求 Python 3.10 及以上。

### 推理服务（线上 / 本地开发）

```bash
python3 -m venv .venv-infer
source .venv-infer/bin/activate
pip3 install -e ".[dev,infer]"
```

仅需 FastAPI 与 ONNX Runtime 相关依赖，**不需要** PyTorch。

### 训练环境

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip3 install -e ".[train]"
# LoRA 微调额外需要：pip3 install peft
# ModelScope 下载额外需要：pip3 install modelscope
```

### ONNX 导出

导出脚本同时需要训练与推理依赖：

```bash
pip3 install -e ".[train,infer]"
```

### 流水线 CLI

推荐在项目虚拟环境中安装（editable 可正常工作）：

```bash
source .venv-infer/bin/activate   # 或先创建：python3 -m venv .venv-infer
pip3 install -e ".[dev]"
flash-echo-pipeline list
```

若使用 Homebrew 全局 `pip3 install -e`，macOS 上可能出现 editable `.pth` 未被解释器加载的情况；本项目 CLI 入口已内置路径修复，重装后可直接运行 `flash-echo-pipeline list`。仍建议日常开发使用上面的 venv。

不安装 package 时的兜底：

```bash
PYTHONPATH=src python3 -m flash_echo_pipeline.cli list
```

## 快速开始（推理服务）

```bash
source .venv-infer/bin/activate
uvicorn flash_echo.app:create_app --factory --reload
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "filler-reply",
    "messages": [{"role": "user", "content": "你怎么看 AI 对教育行业的影响？"}],
    "metadata": {"persona_tag": "male_white_collar", "request_id": "req-001"}
  }'
```

## 模型流水线

项目提供统一流水线入口 `flash-echo-pipeline`，用于封装数据、训练、导出、镜像构建和推理步骤。现有 `tools/` 脚本仍然保留，流水线会读取 `configs/pipelines/*.yaml` 和 `configs/runtimes/*.yaml` 来生成实际执行命令。

### 查看可用流程

```bash
flash-echo-pipeline list
```

当前内置模型：

- `minimind3`
- `qwen3_5_0_8b`

### 推荐执行顺序

一条完整的模型产出链路按下面顺序执行：

```text
data -> download -> train -> upload_checkpoint -> export -> upload_deploy -> infer
```

也可以使用 `full` 一次串起数据、训练、导出、镜像构建和推理服务启动。日常更推荐拆开执行，便于复用已有数据、checkpoint 或 deploy bundle。

### Dry Run

建议先用 dry run 查看即将执行的命令，不会实际下载模型、构建镜像或启动训练：

```bash
flash-echo-pipeline run minimind3.data \
  --persona male_white_collar \
  --dry-run

flash-echo-pipeline run minimind3.train \
  --persona male_white_collar \
  --dry-run

flash-echo-pipeline run qwen3_5_0_8b.export \
  --dry-run
```

### MiniMind3 流程

MiniMind3 使用 `data/filler_prefix/<persona>/` 下的 `train.jsonl`、`valid.jsonl`、`test.jsonl` 进行 SFT 微调。默认产物：

```text
models/checkpoints/minimind3-filler-<persona>-<version>/best
models/deploy/minimind3-filler-<persona-dash>-<version>/
```

`<version>` 来自 `configs/pipelines/minimind3.yaml` 的 `version` 字段，默认是 `v1.0.0`。

数据准备：

```bash
flash-echo-pipeline run minimind3.data \
  --persona male_white_collar \
  --runtime docker.cpu
```

下载预训练模型：

```bash
flash-echo-pipeline run minimind3.download \
  --runtime docker.cpu \
  --resume
```

训练：

```bash
flash-echo-pipeline run minimind3.train \
  --persona male_white_collar \
  --resume
```

`minimind3.train` 只执行预训练模型下载和微调。训练镜像需要提前用
`flash-echo-pipeline image build minimind3.train` 构建，或者通过 `full` pipeline
中的 `build_train_image` step 构建。如果预训练模型已经准备好，只想继续训练 step，
可以从 `finetune` 开始：

```bash
flash-echo-pipeline run minimind3.train \
  --persona male_white_collar \
  --from finetune \
  --resume
```

训练后上传 checkpoint：

```bash
flash-echo-pipeline run minimind3.upload_checkpoint \
  --persona male_white_collar
```

导出 ONNX deploy bundle：

```bash
flash-echo-pipeline run minimind3.export \
  --persona male_white_collar \
  --resume
```

`minimind3.export` 只执行 ONNX 导出。导出镜像需要提前用
`flash-echo-pipeline image build minimind3.export` 构建，或者通过 `full` pipeline
中的 `build_export_image` step 构建。

导出后上传 deploy bundle：

```bash
flash-echo-pipeline run minimind3.upload_deploy \
  --persona male_white_collar
```

启动推理服务：

```bash
flash-echo-pipeline run minimind3.infer \
  --persona male_white_collar \
  --runtime docker.minimind3-infer
```

`minimind3.infer` 只启动推理服务。推理镜像需要提前用
`flash-echo-pipeline image build minimind3.infer` 构建，或者通过 `full` pipeline
中的 `build_infer_image` step 构建。

完整流程：

```bash
flash-echo-pipeline run minimind3.full \
  --persona male_white_collar \
  --resume
```

### Qwen3.5-0.8B 流程

Qwen3.5-0.8B 也支持使用 `data/filler_prefix/<persona>/` 下的 `train.jsonl`、`valid.jsonl`、`test.jsonl` 进行 SFT 微调。默认训练产物会保存到：

```text
models/checkpoints/qwen3_5_0_8b-filler-<persona>-<version>/best
models/deploy/qwen3_5_0_8b-filler-<persona-dash>-<version>-onnx/
```

`<version>` 来自 `configs/pipelines/qwen3_5_0_8b.yaml` 的 `version` 字段，默认是 `v1.0.0`。

导出阶段默认读取 best checkpoint，并输出 persona 专属 ONNX / ORT GenAI deploy bundle。

数据准备：

```bash
flash-echo-pipeline run qwen3_5_0_8b.data \
  --persona male_white_collar \
  --runtime docker.cpu
```

下载预训练模型：

```bash
flash-echo-pipeline run qwen3_5_0_8b.download \
  --runtime docker.qwen-export \
  --resume
```

训练：

```bash
flash-echo-pipeline run qwen3_5_0_8b.train \
  --persona male_white_collar \
  --resume
```

`qwen3_5_0_8b.train` 只执行预训练模型下载和微调。训练镜像需要提前用
`flash-echo-pipeline image build qwen3_5_0_8b.train` 构建，或者通过 `full` pipeline
中的 `build_train_image` step 构建。如果预训练模型已经准备好，只想继续训练 step，
可以从 `finetune` 开始：

```bash
flash-echo-pipeline run qwen3_5_0_8b.train \
  --persona male_white_collar \
  --from finetune \
  --resume
```

训练后上传 checkpoint：

```bash
flash-echo-pipeline run qwen3_5_0_8b.upload_checkpoint \
  --persona male_white_collar
```

导出：

```bash
flash-echo-pipeline run qwen3_5_0_8b.export \
  --persona male_white_collar \
  --resume
```

`qwen3_5_0_8b.export` 只执行 ONNX / ORT GenAI 导出。导出镜像需要提前用
`flash-echo-pipeline image build qwen3_5_0_8b.export` 构建，或者通过 `full` pipeline
中的 `build_export_image` step 构建。

导出后上传 deploy bundle：

```bash
flash-echo-pipeline run qwen3_5_0_8b.upload_deploy \
  --persona male_white_collar
```

启动推理服务：

```bash
flash-echo-pipeline run qwen3_5_0_8b.infer \
  --persona male_white_collar \
  --runtime docker.qwen-infer
```

`qwen3_5_0_8b.infer` 只启动推理服务。推理镜像需要提前用
`flash-echo-pipeline image build qwen3_5_0_8b.infer` 构建，或者通过 `full` pipeline
中的 `build_infer_image` step 构建。

完整流程：

```bash
flash-echo-pipeline run qwen3_5_0_8b.full \
  --persona male_white_collar \
  --resume
```

### COS 产物同步

微调 checkpoint 和 deploy bundle 可以同步到腾讯云 COS：

```text
https://weights-1305049745.cos.ap-shanghai.myqcloud.com
```

在项目根目录 `.env` 中配置 COS 密钥：

```bash
QCLOUD_SECRET_ID='your-secret-id'
QCLOUD_SECRET_KEY='your-secret-key'
```

COS 远端路径按基础模型、微调版本和 persona 组织：

```text
flash-echo/<base_model>/<version>/<persona>/checkpoint/best/
flash-echo/<base_model>/<version>/<persona>/deploy/
```

示例路径：

```text
flash-echo/minimind3/v1.0.0/male_white_collar/checkpoint/best/
flash-echo/qwen3_5_0_8b/v1.0.0/male_white_collar/deploy/
```

在新机器上恢复 checkpoint：

```bash
flash-echo-pipeline run minimind3.download_checkpoint \
  --persona male_white_collar \
  --resume

flash-echo-pipeline run qwen3_5_0_8b.download_checkpoint \
  --persona male_white_collar \
  --resume
```

在新机器上恢复 deploy bundle：

```bash
flash-echo-pipeline run minimind3.download_deploy \
  --persona male_white_collar \
  --resume

flash-echo-pipeline run qwen3_5_0_8b.download_deploy \
  --persona male_white_collar \
  --resume
```

`upload_model` 是 `upload_deploy` 的别名，表示上传推理服务实际消费的模型包。`download_model` 是 `download_deploy` 的别名，适合直接恢复推理所需模型包。

### 构建与推送 Docker 镜像

项目镜像统一使用腾讯云 CCR 单仓库多 tag：

```text
ccr.ccs.tencentyun.com/huyyxy/flash-echo:<tag>
```

登录仓库：

```bash
export TENCENT_CCR_PASSWORD='your-password'
printf '%s\n' "$TENCENT_CCR_PASSWORD" | docker login ccr.ccs.tencentyun.com --username=100011783411 --password-stdin
```

一键构建、拉取或推送全部 x86_64 镜像：

```bash
flash-echo-pipeline image build-all
flash-echo-pipeline image pull-all
flash-echo-pipeline image push-all
```

也可以只操作单个镜像：

```bash
flash-echo-pipeline image build docker.cpu
flash-echo-pipeline image build docker.base-cuda
flash-echo-pipeline image build docker.base-cuda-cu128
flash-echo-pipeline image build minimind3.train
flash-echo-pipeline image build minimind3.export
flash-echo-pipeline image build minimind3.infer
flash-echo-pipeline image build qwen3_5_0_8b.train
flash-echo-pipeline image build qwen3_5_0_8b.train-cu128
flash-echo-pipeline image build qwen3_5_0_8b.export
flash-echo-pipeline image build qwen3_5_0_8b.infer

flash-echo-pipeline image pull qwen3_5_0_8b.infer
flash-echo-pipeline image push qwen3_5_0_8b.infer
```

### 单步执行

流水线支持单步执行，命令中的 `-` 会自动映射到配置里的 `_`：

```bash
flash-echo-pipeline step clean-data \
  --model minimind3 \
  --persona male_white_collar \
  --runtime docker.cpu

flash-echo-pipeline step finetune \
  --model minimind3 \
  --persona male_white_collar \
  --runtime docker.cuda
```

### 断点续跑和强制重跑

`--resume` 会在输出产物已存在时跳过对应 step：

```bash
flash-echo-pipeline run minimind3.train \
  --persona male_white_collar \
  --resume
```

RTX 5090 / Blackwell 机器可使用 CUDA 12.8 训练镜像：

```bash
flash-echo-pipeline image build docker.base-cuda-cu128
flash-echo-pipeline image build qwen3_5_0_8b.train-cu128
flash-echo-pipeline run qwen3_5_0_8b.train-cu128 \
  --persona male_white_collar \
  --resume
```

强制重跑某个 step：

```bash
flash-echo-pipeline run minimind3.full \
  --persona male_white_collar \
  --force finetune
```

从指定 step 开始：

```bash
flash-echo-pipeline run minimind3.full \
  --persona male_white_collar \
  --from finetune
```

### Runtime 选择

常用 runtime：

| Runtime | 用途 |
|---------|------|
| `local` | 本机执行轻量命令 |
| `local.mac` | Mac 本地开发 |
| `docker.cpu` | 数据清洗、审计、CPU 推理或导出 |
| `docker.cuda` | Ubuntu + NVIDIA GPU 训练 |
| `docker.base-cuda-cu128` | CUDA 12.8 / PyTorch cu128 基础镜像，适合 RTX 5090 / Blackwell |
| `docker.minimind3-export` | MiniMind3 ONNX 导出 |
| `docker.minimind3-infer` | MiniMind3 推理服务 |
| `docker.qwen-train` | Qwen3.5-0.8B SFT 训练 |
| `docker.qwen-train-cu128` | Qwen3.5-0.8B SFT 训练，适合 RTX 5090 / Blackwell |
| `docker.qwen-export` | Qwen3.5-0.8B ONNX / ORT GenAI 导出 |
| `docker.qwen-infer` | Qwen3.5-0.8B 推理服务 |

Docker runtime 会默认把项目目录挂载到容器内 `/workspace`。

## 模型集成

推理服务通过 `configs/model_router.v1.json` 将 `persona_tag` 映射到 `models/deploy/<model_version>/` 下的 ONNX bundle。若对应 deploy 包不存在，服务会自动回退到静态规则或 fallback 前缀。

完整训练、导出与部署流程见：

- [docs/training.md](docs/training.md) — MiniMind 数据生成、预训练下载、SFT/LoRA
- [docs/inference.md](docs/inference.md) — ONNX 导出、deploy bundle、服务配置
- [docs/ml_pipeline_requirements.md](docs/ml_pipeline_requirements.md) — 模型流水线需求说明书

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FILLER_CONFIG_DIR` | `configs/` | 运行时配置文件目录 |
| `FILLER_DEPLOY_ROOT` | `models/deploy/` | ONNX 部署包根目录 |
| `FILLER_HOST` | `0.0.0.0` | 服务监听地址 |
| `FILLER_PORT` | `8000` | 服务端口 |
| `QCLOUD_SECRET_ID` | 无 | 腾讯云 COS SecretId，用于模型产物上传/下载 |
| `QCLOUD_SECRET_KEY` | 无 | 腾讯云 COS SecretKey，用于模型产物上传/下载 |
| `QCLOUD_TOKEN` | 无 | 临时密钥 token，可选 |
