# Flash Echo 模型流水线需求说明书

## 1. 背景

Flash Echo 当前同时涉及数据集下载、数据清洗、预训练模型下载、模型微调、测试评估、模型导出、模型上传、训练镜像构建、推理镜像构建和推理服务启动等步骤。项目中已经存在多个可独立执行的脚本，但脚本数量较多，且不同模型、不同环境、不同阶段所需参数并不一致，导致开发和实验过程中需要频繁查阅文档或回忆命令。

同时，开发和运行环境存在明显差异：

- Intel 芯片 MacBook Pro
- Apple Silicon MacBook Air
- Ubuntu + RTX 4060 Ti
- Ubuntu + RTX 4090

训练环境和推理环境也可能依赖不同版本的 PyTorch、Transformers、ONNX Runtime、ONNX Runtime GenAI、FastAPI 等库。因此，模型流水线需要支持通过 Docker 容器运行指定步骤，并能够根据不同机器、芯片架构和 GPU 能力选择合适的运行环境。

本需求说明书旨在定义一个统一的模型流水线系统，使开发者通过少量稳定命令即可完成不同模型、不同 Persona、不同运行环境下的完整业务流程。

## 2. 目标

### 2.1 核心目标

构建一个统一的 Python 流水线入口，用于编排 Flash Echo 模型研发和部署过程中的各类步骤。

目标命令示例：

```bash
flash-echo-pipeline run minimind3.full --persona male_white_collar --runtime docker.cuda --resume
flash-echo-pipeline run qwen3_5_0_8b.export --runtime docker.qwen-export
flash-echo-pipeline step finetune --model minimind3 --persona male_white_collar --runtime docker.cuda
flash-echo-pipeline image build qwen3_5_0_8b.infer
```

流水线系统需要做到：

- 统一管理脚本命令和默认参数
- 支持完整流程、阶段流程和单步执行
- 支持本地运行和 Docker 容器运行
- 支持不同模型的差异化参数和执行逻辑
- 支持训练环境、导出环境、推理环境分离
- 支持输入产物检查、输出产物检查和断点续跑
- 支持 Intel Mac、Apple Silicon Mac 和 Ubuntu GPU 机器
- 支持 MiniMind3 和 Qwen3.5-0.8B 两类模型

### 2.2 非目标

第一阶段不要求重写所有现有脚本。现有 `tools/` 下的训练、导出、推理脚本可以继续作为底层能力存在，流水线系统先负责封装、编排和参数注入。

第一阶段不要求引入复杂的外部工作流系统，如 Airflow、Kubeflow、Prefect 或 MLflow。除非后续需要团队协作、远程调度、实验追踪或集群训练，否则优先保持项目内轻量实现。

## 3. 现有流程

当前业务流程可以抽象为以下阶段：

```text
下载原始数据集
  -> 清洗数据集
  -> 或直接下载清洗后的数据集
  -> 下载预训练模型
  -> 构建训练镜像
  -> 模型微调
  -> 测试评估
  -> 模型上传
  -> 构建推理镜像
  -> 模型推理
```

不同模型不一定都需要完整执行所有步骤。例如：

- MiniMind3 需要按 Persona 准备数据集并进行 SFT 微调。
- Qwen3.5-0.8B 可能更关注预训练模型下载、ONNX 导出、推理镜像构建和推理服务运行。
- 数据清洗步骤可能在本地 CPU 容器中执行。
- 模型微调步骤更适合在 Ubuntu + NVIDIA GPU 机器上执行。
- 推理环境应尽量独立于训练环境，避免推理镜像携带不必要的训练依赖。

## 4. 推荐目录结构

建议新增以下目录和文件：

```text
configs/
  pipelines/
    minimind3.yaml
    qwen3_5_0_8b.yaml
  runtimes/
    local.mac.yaml
    docker.cpu.yaml
    docker.cuda.yaml
    docker.qwen-export.yaml
    docker.qwen-infer.yaml

docker/
  base/
    Dockerfile.cpu
    Dockerfile.cuda
  minimind3/
    Dockerfile.train
    Dockerfile.export
    Dockerfile.infer
  qwen3_5_0_8b/
    Dockerfile.export
    Dockerfile.infer

src/
  flash_echo_pipeline/
    __init__.py
    cli.py
    context.py
    config.py
    runner.py
    runtime/
      __init__.py
      local.py
      docker.py
    steps/
      __init__.py
      data.py
      pretrained.py
      train.py
      evaluate.py
      export.py
      upload.py
      docker_build.py
      infer.py
```

现有目录保留：

```text
src/flash_echo/          线上推理服务
tools/training/          训练与数据脚本 CLI
tools/export/            模型导出脚本 CLI
tools/inference/         推理调试和服务脚本 CLI
configs/               服务运行时配置
data/                  数据集
models/                预训练、checkpoint、deploy bundle
```

后续共享的流水线 step 逻辑沉到 `src/flash_echo_pipeline/steps/`，`tools/` 保留为兼容命令入口。

## 5. 流水线配置

每个模型应有独立的 pipeline 配置文件。配置文件负责描述模型名称、版本、路径、步骤、默认参数和产物检查规则。

示例：`configs/pipelines/minimind3.yaml`

```yaml
model: minimind3
default_persona: male_white_collar
version: v1.0.0

paths:
  raw_data: data/raw/sft_t2t_mini.jsonl
  cleaned_data_dir: data/filler_prefix/{persona}
  pretrained_model: models/pretrained/jingyaogong-minimind-3
  checkpoint: models/checkpoints/minimind3-filler-{persona}-v1.0.0/best
  deploy_dir: models/deploy/minimind3-filler-{persona_dash}-v1.0.0

train:
  epochs: 3
  batch_size: 8
  use_lora: false

pipelines:
  data:
    - download_raw_data
    - clean_data
    - audit_data

  train:
    - download_pretrained
    - finetune
    - evaluate

  export:
    - export_onnx

  infer:
    - build_infer_image
    - serve

  full:
    - download_raw_data
    - clean_data
    - audit_data
    - download_pretrained
    - build_train_image
    - finetune
    - evaluate
    - export_onnx
    - upload_model
    - build_infer_image
    - serve

steps:
  clean_data:
    runtime: docker.cpu
    command:
      - python3
      - tools/training/prepare_sharegpt_filler_prefix.py
      - --persona
      - "{persona}"
    inputs:
      - "{paths.raw_data}"
    outputs:
      - "{paths.cleaned_data_dir}/train.jsonl"
      - "{paths.cleaned_data_dir}/valid.jsonl"
      - "{paths.cleaned_data_dir}/test.jsonl"
    resume: true

  download_pretrained:
    runtime: docker.cpu
    command:
      - python3
      - tools/training/download_minimind3_pretrained.py
    outputs:
      - "{paths.pretrained_model}"
    resume: true

  finetune:
    runtime: docker.cuda
    command:
      - python3
      - tools/training/train_minimind3_sft.py
      - --persona
      - "{persona}"
      - --base-model
      - "{paths.pretrained_model}"
      - --epochs
      - "{train.epochs}"
      - --batch-size
      - "{train.batch_size}"
    inputs:
      - "{paths.cleaned_data_dir}/train.jsonl"
      - "{paths.cleaned_data_dir}/valid.jsonl"
      - "{paths.pretrained_model}"
    outputs:
      - "{paths.checkpoint}/model_card.json"
    resume: true

  export_onnx:
    runtime: docker.cpu
    command:
      - python3
      - tools/export/export_minimind3_onnx.py
      - --persona
      - "{persona}"
    inputs:
      - "{paths.checkpoint}"
    outputs:
      - "{paths.deploy_dir}/model.onnx"
      - "{paths.deploy_dir}/inference_config.json"
    resume: true
```

配置中的 `{persona}`、`{persona_dash}`、`{paths.*}`、`{train.*}` 等变量由流水线系统在运行时解析。

## 6. Runtime 配置

Runtime 用于定义某个 step 在什么环境中执行。每个 step 可以选择 `local` 或 `docker` runtime。

### 6.1 Local Runtime

示例：`configs/runtimes/local.mac.yaml`

```yaml
name: local.mac
type: local
workdir: .
env:
  PYTHONPATH: src
```

Local runtime 适合执行轻量脚本，例如本地数据检查、配置校验或快速推理调试。

### 6.2 Docker CPU Runtime

示例：`configs/runtimes/docker.cpu.yaml`

```yaml
name: docker.cpu
type: docker

image: ccr.ccs.tencentyun.com/huyyxy/flash-echo:base-cpu
dockerfile: docker/base/Dockerfile.cpu
platform: linux/amd64
gpu: none

workdir: /workspace
mounts:
  - source: .
    target: /workspace

env:
  PYTHONPATH: /workspace/src
  HF_HOME: /workspace/models/.cache/huggingface
  TRANSFORMERS_CACHE: /workspace/models/.cache/huggingface
```

Docker CPU runtime 适合执行数据清洗、数据审计、预训练模型下载、CPU 导出和轻量推理调试。

### 6.3 Docker CUDA Runtime

基础 CUDA 镜像示例：`configs/runtimes/docker.base-cuda.yaml`

```yaml
name: docker.base-cuda
type: docker

image: ccr.ccs.tencentyun.com/huyyxy/flash-echo:base-cuda
dockerfile: docker/base/Dockerfile.cuda
platform: linux/amd64
gpu: all

workdir: /workspace
mounts:
  - source: .
    target: /workspace

env:
  PYTHONPATH: /workspace/src
  HF_HOME: /workspace/models/.cache/huggingface
  TRANSFORMERS_CACHE: /workspace/models/.cache/huggingface
```

Docker CUDA runtime 适合在 Ubuntu + RTX 4060 Ti 或 RTX 4090 机器上执行模型微调和 GPU 评估。

模型训练 runtime 可以继承基础 CUDA 镜像，例如 `docker.cuda.yaml` 使用：

```yaml
image: ccr.ccs.tencentyun.com/huyyxy/flash-echo:minimind3-train
dockerfile: docker/minimind3/Dockerfile.train
platform: linux/amd64
gpu: all
```

### 6.4 Qwen Export Runtime

Qwen3.5-0.8B 的导出依赖可能与主服务和 MiniMind3 不一致，应使用独立导出镜像。

示例：`configs/runtimes/docker.qwen-export.yaml`

```yaml
name: docker.qwen-export
type: docker

image: ccr.ccs.tencentyun.com/huyyxy/flash-echo:qwen3_5_0_8b-export
dockerfile: docker/qwen3_5_0_8b/Dockerfile.export
platform: linux/amd64
gpu: none

workdir: /workspace
mounts:
  - source: .
    target: /workspace

env:
  PYTHONPATH: /workspace/src
  HF_HOME: /workspace/models/.cache/huggingface
  ORTGENAI_CACHE: /workspace/models/.cache/onnxruntime-genai
```

### 6.5 Qwen Infer Runtime

Qwen3.5-0.8B 的推理环境应和导出环境分离，推理镜像不应安装不必要的训练和导出依赖。

示例：`configs/runtimes/docker.qwen-infer.yaml`

```yaml
name: docker.qwen-infer
type: docker

image: ccr.ccs.tencentyun.com/huyyxy/flash-echo:qwen3_5_0_8b-infer
dockerfile: docker/qwen3_5_0_8b/Dockerfile.infer
platform: linux/amd64
gpu: none

workdir: /workspace
ports:
  - host: 8080
    container: 8080

mounts:
  - source: .
    target: /workspace

env:
  PYTHONPATH: /workspace/src
  HF_HOME: /workspace/models/.cache/huggingface
  ORTGENAI_CACHE: /workspace/models/.cache/onnxruntime-genai
```

## 7. Step 定义

每个 step 是流水线的最小执行单元。一个 step 至少应包含：

- step 名称
- 运行 runtime
- 执行命令
- 输入产物
- 输出产物
- 是否支持断点续跑
- 是否允许强制重跑

示例：

```yaml
finetune:
  runtime: docker.cuda
  command:
    - python3
    - tools/training/train_minimind3_sft.py
    - --persona
    - "{persona}"
  inputs:
    - "{paths.cleaned_data_dir}/train.jsonl"
    - "{paths.cleaned_data_dir}/valid.jsonl"
  outputs:
    - "{paths.checkpoint}/model_card.json"
  resume: true
```

执行规则：

- 执行前检查 `inputs` 是否存在。
- 如果 `resume=true` 且所有 `outputs` 已存在，则默认跳过。
- 如果用户传入 `--force`，则强制执行指定 step。
- step 失败时，流水线应立即停止，并打印失败 step、执行命令和建议检查项。
- 每个 step 的开始时间、结束时间、耗时、runtime、镜像名和命令应记录到日志。

## 8. Docker 镜像管理

流水线系统需要支持镜像构建、登录后推送命令。镜像统一放入腾讯云 CCR 单仓库多 tag：

```text
ccr.ccs.tencentyun.com/huyyxy/flash-echo:<tag>
```

登录命令不应把密码写入仓库文件，应通过环境变量或交互方式输入：

```bash
printf '%s\n' "$TENCENT_CCR_PASSWORD" | docker login ccr.ccs.tencentyun.com --username=100011783411 --password-stdin
```

全量构建与推送：

```bash
flash-echo-pipeline image build-all
flash-echo-pipeline image push-all
```

单镜像构建命令：

```bash
flash-echo-pipeline image build docker.cpu
flash-echo-pipeline image build docker.base-cuda
flash-echo-pipeline image build minimind3.train
flash-echo-pipeline image build minimind3.export
flash-echo-pipeline image build minimind3.infer
flash-echo-pipeline image build qwen3_5_0_8b.train
flash-echo-pipeline image build qwen3_5_0_8b.export
flash-echo-pipeline image build qwen3_5_0_8b.infer
```

镜像构建应从 runtime 配置读取：

- image tag
- Dockerfile 路径
- build context
- platform
- build args

构建前应检查 Dockerfile 是否存在。构建完成后应检查镜像是否存在。

推荐 Dockerfile 命名：

```text
docker/
  minimind3/
    Dockerfile.train
    Dockerfile.export
    Dockerfile.infer
  qwen3_5_0_8b/
    Dockerfile.export
    Dockerfile.infer
  base/
    Dockerfile.cpu
    Dockerfile.cuda
```

## 9. 命令行设计

### 9.1 查看可用流水线

```bash
flash-echo-pipeline list
flash-echo-pipeline list --model minimind3
```

### 9.2 运行完整流程

```bash
flash-echo-pipeline run minimind3.full --persona male_white_collar --runtime docker.cuda --resume
```

### 9.3 运行阶段流程

```bash
flash-echo-pipeline run minimind3.data --persona male_white_collar --runtime docker.cpu
flash-echo-pipeline run minimind3.train --persona male_white_collar --runtime docker.cuda
flash-echo-pipeline run minimind3.export --persona male_white_collar
flash-echo-pipeline run qwen3_5_0_8b.export --runtime docker.qwen-export
```

### 9.4 运行单个步骤

```bash
flash-echo-pipeline step clean-data --model minimind3 --persona male_white_collar --runtime docker.cpu
flash-echo-pipeline step finetune --model minimind3 --persona male_white_collar --runtime docker.cuda
flash-echo-pipeline step infer --model qwen3_5_0_8b --runtime docker.qwen-infer
```

### 9.5 构建与推送镜像

```bash
flash-echo-pipeline image build-all
flash-echo-pipeline image push-all

flash-echo-pipeline image build qwen3_5_0_8b.export
flash-echo-pipeline image push qwen3_5_0_8b.export
```

### 9.6 Dry Run

流水线系统应支持 dry run，用于只打印将要执行的命令，不实际运行。

```bash
flash-echo-pipeline run minimind3.full --persona male_white_collar --runtime docker.cuda --dry-run
```

Dry run 应输出：

- 展开后的 step 列表
- 每个 step 使用的 runtime
- 每个 step 的输入和输出
- 实际执行命令
- Docker run 参数

## 10. 断点续跑与强制执行

流水线应支持断点续跑：

```bash
flash-echo-pipeline run minimind3.full --persona male_white_collar --resume
```

当 step 的全部输出产物已存在时，默认跳过该 step。

用户可以强制执行某个或多个步骤：

```bash
flash-echo-pipeline run minimind3.full --persona male_white_collar --force clean_data
flash-echo-pipeline run minimind3.full --persona male_white_collar --force finetune --force export_onnx
```

用户也可以从指定步骤开始：

```bash
flash-echo-pipeline run minimind3.full --persona male_white_collar --from finetune
```

## 11. 模型差异适配

### 11.1 MiniMind3

MiniMind3 流程重点包括：

- 下载 MiniMind3 预训练模型
- 按 Persona 清洗和生成 filler prefix 数据集
- SFT 或 LoRA 微调
- 测试集评估
- 导出 ONNX deploy bundle
- 更新或验证 `configs/model_router.v1.json`
- 启动 Flash Echo 推理服务

Persona 命名需要处理两种形式：

- 数据和 checkpoint 使用下划线，例如 `male_white_collar`
- deploy 和 model version 使用连字符，例如 `male-white-collar`

流水线系统应自动提供 `{persona}` 和 `{persona_dash}` 两个变量。

### 11.2 Qwen3.5-0.8B

Qwen3.5-0.8B 流程重点包括：

- 下载 Hugging Face 预训练模型
- 构建导出镜像
- 导出 ONNX 或 ONNX Runtime GenAI bundle
- 构建推理镜像
- 启动本地 OpenAI 兼容推理服务

Qwen3.5-0.8B 的 Transformers 和 ONNX Runtime GenAI 版本可能与 Flash Echo 主服务不兼容，因此必须允许其使用独立 Dockerfile 和独立 runtime 配置。

## 12. 日志与状态记录

每次流水线运行应生成运行记录，建议保存到：

```text
runs/
  <timestamp>-<pipeline-name>/
    run.yaml
    logs/
      clean_data.log
      finetune.log
      export_onnx.log
```

运行记录应包含：

- pipeline 名称
- model
- persona
- runtime
- git commit
- 执行机器信息
- step 列表
- 每个 step 的开始时间、结束时间和状态
- 每个 step 的输入输出产物
- 展开后的命令

`runs/` 默认不进入 git。

## 13. 产物管理

推荐产物目录：

```text
data/raw/                       原始数据
data/filler_prefix/<persona>/   清洗后的 SFT 数据
models/pretrained/              预训练模型
models/checkpoints/             微调 checkpoint
models/deploy/                  推理 deploy bundle
runs/                           流水线运行记录
```

产物检查规则：

- 数据清洗完成后必须存在 `train.jsonl`、`valid.jsonl`、`test.jsonl`
- 训练完成后必须存在 `model_card.json`
- ONNX 导出完成后必须存在 `model.onnx` 和 `inference_config.json`
- 推理服务启动前必须检查 deploy bundle 是否存在
- 上传模型前必须检查目标产物完整性

## 14. 安全与可移植性要求

Docker 执行时应遵守以下原则：

- 默认只挂载项目目录到容器内 `/workspace`
- 不把宿主机敏感目录挂载进容器
- Token、密钥和 registry 登录信息通过环境变量或宿主机已有 Docker 登录态传入
- 不把 `.env`、缓存、模型大文件和运行日志提交到 git
- Apple Silicon 默认使用 `linux/arm64` 或显式指定可兼容的 `linux/amd64`
- CUDA runtime 只在 Linux + NVIDIA GPU 环境使用

## 15. 分阶段落地计划

### 阶段一：最小可用流水线

实现目标：

- 新增 `flash-echo-pipeline` CLI
- 支持读取 pipeline YAML 和 runtime YAML
- 支持 local runtime
- 支持 Docker runtime
- 支持 dry run
- 支持输入输出检查
- 支持 `--resume`
- 先封装 MiniMind3 的 `clean_data`、`download_pretrained`、`finetune`、`export_onnx`

### 阶段二：Docker 镜像管理

实现目标：

- 整理 Dockerfile 到 `docker/` 目录
- 支持 `flash-echo-pipeline image build ...`
- 支持 CPU、CUDA、Qwen export、Qwen infer runtime
- 支持 Docker platform、ports、mounts、env 配置

### 阶段三：完善模型流程

实现目标：

- 补齐 MiniMind3 的 `evaluate`、`upload_model`、`serve`
- 接入 Qwen3.5-0.8B 的下载、导出、推理流程
- 支持按模型选择默认 runtime
- 支持从指定 step 开始执行

### 阶段四：可观测性和实验管理

实现目标：

- 生成 `runs/<timestamp>/` 运行记录
- 记录每个 step 的日志
- 记录 git commit 和机器信息
- 支持查看最近运行状态
- 支持失败后从失败 step 恢复

## 16. 验收标准

第一版流水线完成后，应满足以下标准：

- 可以通过单条命令 dry run 展示 MiniMind3 完整流程
- 可以通过 Docker CPU runtime 执行数据清洗
- 可以通过 Docker CUDA runtime 执行 MiniMind3 微调
- 可以通过 Docker runtime 执行 ONNX 导出
- 可以在输出产物已存在时自动跳过 step
- 可以通过 `--force` 重跑指定 step
- 可以构建 Qwen3.5-0.8B export 和 infer 镜像
- 命令、runtime、输入输出产物均能在日志中追踪

## 17. 总结

本项目的规范化重点不应只是增加格式化工具或整理脚本文件名，而是建立一套可复用、可配置、可容器化的模型流水线系统。

通过将现有脚本封装为 pipeline step，并为每个 step 配置 runtime、Docker 镜像、输入输出和断点续跑规则，可以显著降低命令记忆成本，同时让 MiniMind3 和 Qwen3.5-0.8B 在不同机器和不同依赖环境下保持可复现执行。
