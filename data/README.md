# 数据目录

版本化数据集请使用以下目录结构：

```text
raw/          未标注或原始 query 语料
processed/    train.jsonl, valid.jsonl, test.jsonl
hard_cases/   每次模型迭代固定使用的回归样本
```

每条 JSONL 记录应遵循如下格式：

```json
{"query":"帮我写一封请假邮件","trigger":1,"filler_type":"ACKNOWLEDGE","source":"task_dialog","label_method":"llm_reviewed"}
```
