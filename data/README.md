# Data Directory

Use the following layout for versioned datasets:

```text
raw/          Unlabeled or source query corpora
processed/    train.jsonl, valid.jsonl, test.jsonl
hard_cases/   Fixed regression samples for each model iteration
```

Each JSONL record should follow:

```json
{"query":"帮我写一封请假邮件","trigger":1,"filler_type":"ACKNOWLEDGE","source":"task_dialog","label_method":"llm_reviewed"}
```
