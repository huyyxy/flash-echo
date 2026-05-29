# MiniMind3 训练流程

本文档描述 Flash Echo 离线训练环境的使用方式。训练环境依赖 PyTorch 与 Transformers，**不需要** ONNX Runtime。

## 环境准备

```bash
python -m venv .venv-train
source .venv-train/bin/activate
pip install -e ".[train]"
```

可选依赖：

- LoRA 微调：`pip install peft`
- ModelScope 下载：`pip install modelscope`

## 数据准备

参见 [data/README.md](../data/README.md)。简要步骤：

1. 下载原始 ShareGPT 语料到 `data/raw/sft_t2t_mini.jsonl`
2. 运行 `tools/training/prepare_sharegpt_filler_prefix.py` 生成 `data/filler_prefix/<persona>/`

## 下载 MiniMind3 预训练模型

训练默认使用 `jingyaogong/minimind-3`（Transformers / Safetensors 格式）。MiniMind3 基于 Qwen3 架构，需 **Transformers 4.51 及以上、5 以下**；5.x 会在 import 时依赖 PyTorch 2.6+ 的 API，与常见的 PyTorch 2.4 环境不兼容。

```bash
pip install -e ".[train]"
python3 tools/training/download_minimind3_pretrained.py
```

默认保存到 `models/pretrained/jingyaogong-minimind-3`。

国内可用 ModelScope 镜像：

```bash
pip install modelscope
python3 tools/training/download_minimind3_pretrained.py --source modelscope
```

或使用 Hugging Face 镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 tools/training/download_minimind3_pretrained.py
```

若需要刷新本地副本，加上 `--overwrite`。

## SFT 微调

准备好 `data/filler_prefix/<persona>/train.jsonl`、`valid.jsonl` 后执行：

```bash
python3 tools/training/train_minimind3_sft.py \
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
python3 tools/training/train_minimind3_sft.py \
  --persona male_white_collar \
  --use-lora --lora-r 16
```

## 产物目录

| 目录 | 说明 |
|------|------|
| `models/pretrained/` | 预训练基座模型（训练输入） |
| `models/checkpoints/` | SFT/LoRA checkpoint（训练输出，含下划线 persona） |

训练完成后，使用 [docs/inference.md](inference.md) 中的导出脚本将 checkpoint 转为 ONNX deploy bundle。
