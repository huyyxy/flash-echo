# 数据目录

版本化数据集请使用以下目录结构：

```text
raw/          未标注或原始 query 语料
processed/    train.jsonl, valid.jsonl, test.jsonl
hard_cases/   每次模型迭代固定使用的回归样本
```

## 下载原始语料

MiniMind ShareGPT 格式的 `sft_t2t_mini.jsonl`（供 `scripts/prepare_sharegpt_jsonl.py` 使用）：

```bash
mkdir -p data/raw
curl -L -o data/raw/sft_t2t_mini.jsonl \
  https://datasets-1305049745.cos.ap-shanghai.myqcloud.com/minimind/sft_t2t_mini.jsonl
```

## 预处理为训练集

在项目根目录执行 `scripts/prepare_sharegpt_jsonl.py`：从 ShareGPT 语料的 `conversations` 中抽取用户轮次作为 `query`，经 OpenAI 兼容 Chat API 打标后写入划分文件。

默认路径：

| 方向 | 路径 |
|------|------|
| 输入 | `data/raw/sft_t2t_mini.jsonl` |
| 输出 | `data/processed/train.jsonl` |
| 输出 | `data/processed/valid.jsonl` |
| 输出 | `data/processed/test.jsonl` |

可用 `--input`、`--output-dir` 覆盖；例如 `--output-dir data/processed/v2` 会生成 `data/processed/v2/train.jsonl` 等三个文件。

需配置 API（环境变量、项目根目录 `.env` 或命令行参数）：

- `OPENAI_API_KEY`：密钥（也可用 `--api-key`）
- `OPENAI_BASE_URL`：兼容网关根地址，默认 `https://api.openai.com/v1`
- `OPENAI_MODEL`：模型名（也可用 `--model`）

配置优先级为：命令行参数 > 当前进程环境变量 > 项目根目录 `.env` > 内置默认值。

项目根目录 `.env` 示例：

```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

默认全量处理（读取 `data/raw/sft_t2t_mini.jsonl`，写入 `data/processed/{train,valid,test}.jsonl`）：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"

python scripts/prepare_sharegpt_jsonl.py \
  --input data/raw/sft_t2t_mini.jsonl \
  --output-dir data/processed
```

使用阿里云百炼（DashScope OpenAI 兼容接口）：

```bash
export OPENAI_API_KEY="your-dashscope-api-key"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen-plus"

python scripts/prepare_sharegpt_jsonl.py \
  --base-url "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL"
```

小规模试跑（最多打标 50 条，覆盖已有输出）：

```bash
python scripts/prepare_sharegpt_jsonl.py --max-records 50 --overwrite
```

默认支持断点续跑：已出现在 `data/processed/train.jsonl`、`valid.jsonl`、`test.jsonl` 中的 `query` 会跳过。若要重新生成这三个文件，加上 `--overwrite`。

更多参数见脚本内文档：`python scripts/prepare_sharegpt_jsonl.py --help`。

## 下载预训练模型

训练默认使用 `hfl/rbt3`。建议先下载到本地，避免训练时重复联网：

```bash
pip install -e ".[ml]"
python scripts/download_pretrained_model.py \
  --model hfl/rbt3 \
  --output-dir models/pretrained/hfl-rbt3
```

若需要刷新本地副本，加上 `--overwrite`。

## 训练分类模型

准备好 `data/processed/train.jsonl`、`valid.jsonl`、`test.jsonl` 后执行：

```bash
python scripts/train.py \
  --base-model models/pretrained/hfl-rbt3 \
  --output-dir models/checkpoints/filler-cls-v1 \
  --epochs 3 \
  --batch-size 32
```

默认会保存验证集 `type_macro_f1` 最优的 checkpoint 到
`models/checkpoints/filler-cls-v1/best/`，其中包含 tokenizer、模型权重、
`labels.json` 和 `model_card.json`。如果没有传 `--base-model`，脚本会优先使用
`models/pretrained/hfl-rbt3`，不存在时回退到在线模型名 `hfl/rbt3`。

## 处理后记录格式

每条 JSONL 记录应遵循如下格式：

```json
{"query":"帮我写一封请假邮件","trigger":1,"filler_type":"ACKNOWLEDGE","source":"task_dialog","label_method":"llm_reviewed"}
```
