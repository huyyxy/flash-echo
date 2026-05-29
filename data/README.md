# 数据目录

版本化数据集请使用以下目录结构：

```text
raw/                        未标注或原始 ShareGPT 语料
filler_prefix/<persona>/    MiniMind3 短垫话前缀 SFT 数据（按 Persona 分目录）
hard_cases/                 每次模型迭代固定使用的回归样本（不参与默认训练流水线）
```

支持的 Persona（slug 使用下划线）：`male_white_collar`、`female_receptionist`、`grandpa`、`young_girl`。

命名约定：

| 场景 | 示例（`male_white_collar`） |
|------|-----------------------------|
| 数据目录 | `data/filler_prefix/male_white_collar/` |
| 训练 checkpoint | `models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/` |
| 线上部署包 | `models/deploy/minimind3-filler-male-white-collar-v1.0.0/` |

Persona slug 在路径中保留下划线；部署目录与 `model_version` 将下划线替换为连字符，以便与 Flash Echo 服务默认路由一致。

## 下载原始语料

MiniMind ShareGPT 格式的 `sft_t2t_mini.jsonl`（供 `scripts/prepare_sharegpt_filler_prefix.py` 使用）：

```bash
mkdir -p data/raw
curl -L -o data/raw/sft_t2t_mini.jsonl \
  https://datasets-1305049745.cos.ap-shanghai.myqcloud.com/minimind/sft_t2t_mini.jsonl
```

## 预处理为 MiniMind3 SFT 训练集

在项目根目录执行 `scripts/prepare_sharegpt_filler_prefix.py`：从 ShareGPT 语料的 `conversations` 中抽取用户轮次作为 `query`，经 OpenAI 兼容 Chat API 生成对应 Persona 风格的 `filler_prefix` 后，按稳定哈希划分并写入 `train.jsonl` / `valid.jsonl` / `test.jsonl`。

每个 Persona 需单独跑一遍（`--persona` 不同，输出目录不同）。

默认路径：

| 方向 | 路径 |
|------|------|
| 输入 | `data/raw/sft_t2t_mini.jsonl` |
| 输出 | `data/filler_prefix/<persona>/train.jsonl` |
| 输出 | `data/filler_prefix/<persona>/valid.jsonl` |
| 输出 | `data/filler_prefix/<persona>/test.jsonl` |

可用 `--input`、`--output-dir`、`--persona` 覆盖；例如 `--persona female_receptionist` 会写入 `data/filler_prefix/female_receptionist/` 下的三个文件。

需配置 API（环境变量、项目根目录 `.env` 或命令行参数）：

- `OPENAI_API_KEY`：密钥（必填；也可用 `--api-key`）
- `OPENAI_BASE_URL`：兼容网关根地址，默认 `https://api.openai.com/v1`
- `OPENAI_MODEL`：模型名（必填；也可用 `--model`）

配置优先级为：命令行参数 > 当前进程环境变量 > 项目根目录 `.env` > 内置默认值。

项目根目录 `.env` 示例：

```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

默认全量处理（读取 `data/raw/sft_t2t_mini.jsonl`，写入 `data/filler_prefix/male_white_collar/{train,valid,test}.jsonl`）：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"

python3 scripts/prepare_sharegpt_filler_prefix.py \
  --input data/raw/sft_t2t_mini.jsonl \
  --persona male_white_collar
```

使用阿里云百炼（DashScope OpenAI 兼容接口）：

```bash
export OPENAI_API_KEY="your-dashscope-api-key"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen-plus"

python3 scripts/prepare_sharegpt_filler_prefix.py \
  --base-url "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL" \
  --persona male_white_collar
```

小规模试跑（最多生成 50 条，覆盖已有输出）：

```bash
python3 scripts/prepare_sharegpt_filler_prefix.py --max-records 50 --overwrite
```

默认支持断点续跑：已写入 `output-dir` 下三个划分文件中的 `query` 会跳过。若要重新生成，加上 `--overwrite`。可用 `--workers` 提高 LLM 并发（文件写入仍在主线程）。

更多参数见：`python3 scripts/prepare_sharegpt_filler_prefix.py --help`。

后台长时间运行示例：

```bash
PYTHONUNBUFFERED=1 nohup python3 -u scripts/prepare_sharegpt_filler_prefix.py \
  --persona male_white_collar \
  > prepare_filler_prefix.log 2>&1 &
```

## 下载 MiniMind3 预训练模型

训练默认使用 `jingyaogong/minimind-3`（Transformers / Safetensors 格式）。MiniMind3 基于 Qwen3 架构，需 **Transformers 4.51 及以上、5 以下**（例如 4.57.6）；5.x 会在 import 时依赖 PyTorch 2.6+ 的 API，与常见的 PyTorch 2.4 环境不兼容。建议先下载到本地，避免训练时重复联网：

```bash
pip3 install -e ".[ml]"
python3 scripts/download_minimind3_pretrained.py
```

默认保存到 `models/pretrained/jingyaogong-minimind-3`。国内可用 ModelScope 镜像：

```bash
pip3 install modelscope
python3 scripts/download_minimind3_pretrained.py --source modelscope
```

或使用 Hugging Face 镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 scripts/download_minimind3_pretrained.py
```

若需要刷新本地副本，加上 `--overwrite`。

## 训练 MiniMind3 短垫话前缀模型

准备好 `data/filler_prefix/<persona>/train.jsonl`、`valid.jsonl` 后执行（`--train` / `--valid` 可省略，会按 `--persona` 自动解析）：

```bash
pip3 install -e ".[ml]"

python3 scripts/train_minimind3_sft.py \
  --persona male_white_collar \
  --base-model models/pretrained/jingyaogong-minimind-3 \
  --epochs 3 \
  --batch-size 8
```

默认会保存验证集 loss 最优的 checkpoint 到
`models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/best/`，其中包含 tokenizer、
模型权重和 `model_card.json`。如果没有传 `--base-model`，脚本会优先使用
`models/pretrained/jingyaogong-minimind-3`，不存在时回退到在线模型名 `jingyaogong/minimind-3`。

LoRA 微调（需额外 `pip install peft`）：

```bash
python3 scripts/train_minimind3_sft.py \
  --persona male_white_collar \
  --use-lora --lora-r 16
```

## 导出 ONNX 部署包

SFT 完成后，可将 best checkpoint 导出为线上推理用的 ONNX 包。导出时请显式指定连字符形式的 `model_version` 与 `output-dir`，以便 Flash Echo 服务在 `models/deploy/` 下自动发现：

```bash
pip3 install -e ".[ml]"

python3 scripts/export_minimind3_onnx.py \
  --persona male_white_collar \
  --checkpoint models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/best \
  --model-version minimind3-filler-male-white-collar-v1.0.0 \
  --output-dir models/deploy/minimind3-filler-male-white-collar-v1.0.0
```

若 checkpoint 与输出目录均使用脚本默认路径，也可只传 `--persona`；但默认部署目录名含下划线，服务不会自动加载，需手动指定上述连字符路径或调整路由配置。

LoRA checkpoint 会在导出前自动合并；必要时加上 `--base-model models/pretrained/jingyaogong-minimind-3`。

## 处理后记录格式

每条 JSONL 记录为 MiniMind3 ShareGPT SFT 格式，包含 `conversations` 数组：

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "用户：帮我写一封请假邮件\n请生成一句可续写的短垫话前缀："
    },
    {
      "role": "assistant",
      "content": "好的，我来帮你整理一下，"
    }
  ]
}
```

训练时损失仅在 `assistant` 回复 token 上计算；Persona 风格由数据目录和独立模型版本体现，不作为单次推理的文本输入参数。
