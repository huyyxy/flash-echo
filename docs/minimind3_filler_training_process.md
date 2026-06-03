# MiniMind3 短前缀生成模型训练流程

> **方案 A：训练时带标签增强，推理时仅输入 Query + Persona**

本文档描述实时语音交互场景中 MiniMind3 垫话 / 过渡回复模型的完整训练流程，涵盖数据设计、标注规范、训练阶段、推理配置与评估方法。Flash Echo 工程化训练命令见 [training.md](training.md)。

---

## 1. 目标

本模型用于实时语音交互场景中的**垫话 / 过渡回复**生成。

在用户说话结束后、Main LLM 尚未返回首 token 前，系统调用 MiniMind3 短前缀模型，根据用户 Query 和当前 Persona 生成一句自然、简短、安全、可被主模型继续接上的过渡语。

**示例：**

| 字段 | 内容 |
|------|------|
| 用户 | 我想做一个实时语音助手，怎么降低首字延迟？ |
| Persona | 专业技术顾问 |
| 模型输出 | 我先按链路帮你拆。 |

该模型**不负责正式回答问题**，只负责生成一句承接语。

---

## 2. 训练与推理差异

### 2.1 训练阶段输入

训练时样本包含完整标签：

```xml
<persona>专业技术顾问</persona>
<intent>technical_advice</intent>
<risk>normal</risk>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
<filler>我先按链路帮你拆。</filler>
```

其中：

| 字段 | 说明 |
|------|------|
| `persona` | 助理角色风格 |
| `intent` | 用户输入意图 |
| `risk` | 风险等级 |
| `query` | 用户真实输入 |
| `filler` | 目标输出 |

### 2.2 推理阶段输入

真实对话时，系统不要求提供 `intent` 和 `risk`，只输入：

```xml
<persona>专业技术顾问</persona>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

模型输出：

```
我先按链路帮你拆。
```

### 2.3 设计原因

训练时加入 `intent` 和 `risk` 的目的**不是**让线上系统依赖这些字段，而是帮助模型在训练早期建立更清晰的映射关系：

| Query 类型 | 承接风格 |
|-----------|---------|
| 技术咨询类 | 结构化承接 |
| 情绪倾诉类 | 温和承接 |
| 高风险 Query | 谨慎承接 |
| 投诉类 | 客服式承接 |

随后通过大量**无标签样本**，让模型适应线上真实输入格式：`persona + query → filler`。

因此训练集需要同时包含：

- **带标签样本**：`persona + intent + risk + query → filler`
- **去标签样本**：`persona + query → filler`

---

## 3. 模型任务定义

### 3.1 标准推理输入

```xml
<persona>{persona}</persona>
<query>{query}</query>
```

**示例：**

```xml
<persona>专业技术顾问</persona>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

### 3.2 标准输出

输出只能是一句中文短前缀 `{filler}`。

### 3.3 输出要求

输出必须满足：

1. 只能一句话
2. 中文 **6 到 18 个字**为佳，最长不超过 **24 个字**
3. 只承接，不回答具体问题
4. 不包含事实判断
5. 不做医疗、法律、金融等风险结论
6. 不承诺结果
7. 不反问用户
8. 语气符合 Persona
9. 主模型可以从这句话之后自然接着回答

### 3.4 正例

```
我先按链路帮你拆。
我先帮你梳理重点。
这个我会谨慎看。
我先按步骤帮你看。
我们一步步来看。
我先理解一下你的问题。
```

### 3.5 反例

```
你可以通过流式推理和缓存降低延迟。
这个问题很简单。
我保证帮你解决。
你是不是想问模型推理优化？
别担心，肯定没事。
建议你马上去医院。
```

---

## 4. 数据集总体设计

训练数据分为四类：

| 数据类型 | 作用 | 推理时是否出现 |
|---------|------|--------------|
| Full-tag SFT 数据 | 学习 intent / risk / persona 与 filler 的关系 | 否 |
| No-tag SFT 数据 | 适应线上真实输入格式 | 是 |
| Mixed-tag SFT 数据 | 增强字段缺失鲁棒性 | 部分 |
| Preference 数据 | 偏好更短、更安全、更自然的 filler | 否 |

**推荐比例：**

| 场景 | No-tag | Full-tag | Mixed-tag |
|------|--------|----------|-----------|
| 通用 | 50% | 30% | 20% |
| 线上一定只有 persona + query | **60%–70%** | 20%–30% | 10%–20% |
| 精确推荐（仅 persona + query） | **60%** | **25%** | **15%** |

---

## 5. 数据格式

### 5.1 Full-tag 样本

```json
{
  "input": "<persona>专业技术顾问</persona>\n<intent>technical_advice</intent>\n<risk>normal</risk>\n<query>我想做一个实时语音助手，怎么降低首字延迟？</query>",
  "output": "我先按链路帮你拆。"
}
```

### 5.2 No-tag 样本

```json
{
  "input": "<persona>专业技术顾问</persona>\n<query>我想做一个实时语音助手，怎么降低首字延迟？</query>",
  "output": "我先按链路帮你拆。"
}
```

### 5.3 Mixed-tag 样本

只保留部分标签：

```json
{
  "input": "<persona>专业技术顾问</persona>\n<intent>technical_advice</intent>\n<query>我想做一个实时语音助手，怎么降低首字延迟？</query>",
  "output": "我先按链路帮你拆。"
}
```

或：

```json
{
  "input": "<persona>专业技术顾问</persona>\n<risk>normal</risk>\n<query>我想做一个实时语音助手，怎么降低首字延迟？</query>",
  "output": "我先按链路帮你拆。"
}
```

### 5.4 Chat SFT 格式

如果使用 MiniMind 的对话 SFT 格式，可转为：

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "<persona>专业技术顾问</persona>\n<query>我想做一个实时语音助手，怎么降低首字延迟？</query>"
    },
    {
      "role": "assistant",
      "content": "我先按链路帮你拆。"
    }
  ]
}
```

对于小模型，建议保持 prompt 极短，**不要使用很长的 system prompt**。

---

## 6. Persona 体系设计

### 6.1 推荐 Persona

第一版不宜过多，建议 **5 到 8 个**：

- 通用助理
- 专业技术顾问
- 智能客服
- 学习教练
- 温和陪伴
- 销售顾问
- 健康信息助理
- 政务 / 办事助手

### 6.2 Persona 风格示例

同一个 Query：**「我想提高英语口语，怎么练？」**

| Persona | 输出 |
|---------|------|
| 专业技术顾问 | 我先按方法帮你拆。 |
| 学习教练 | 我们一步步来看。 |
| 温和陪伴 | 好呀，我们慢慢来。 |
| 通用助理 | 我先帮你梳理一下。 |

### 6.3 Persona 数据要求

每个 Persona 都需要覆盖主要 intent 和 risk，不要只在少数场景里出现。否则模型容易出现：

- 看到「专业技术顾问」只会输出「我先帮你拆」
- 看到「温和陪伴」只会输出「我在听」

---

## 7. Intent 体系设计

虽然推理时不输入 intent，但训练阶段仍建议标注 intent。

### 7.1 第一版 Intent 列表（12–16 个）

```
greeting          thanks           closing          ack_only
unclear           technical_advice general_advice   fact_question
task_request      customer_complaint order_status   learning_help
emotional_support health_related   legal_related    finance_related
```

### 7.2 Intent 与 filler 风格

| Intent | 推荐 filler 风格 |
|--------|----------------|
| `technical_advice` | 拆解、梳理、分步骤 |
| `emotional_support` | 倾听、理解、陪伴 |
| `customer_complaint` | 受理、核对、处理 |
| `health_related` | 谨慎、认真、先了解情况 |
| `learning_help` | 一步步、先看思路 |
| `task_request` | 直接确认开始处理 |

---

## 8. Risk 体系设计

### 8.1 推荐 Risk 标签

```
normal | sensitive | high
```

### 8.2 Risk 定义

| Risk | 含义 | 示例 |
|------|------|------|
| `normal` | 普通问题 | 技术、学习、生活建议 |
| `sensitive` | 可能涉及个人、医疗、法律、金融等 | 睡不好、投资、合同 |
| `high` | 可能涉及急症、违法、自伤、重大财务风险等 | 胸口疼、轻生、被骗转账 |

### 8.3 高风险样本输出原则

高风险样本必须训练模型更克制。

**示例：**

```xml
<persona>健康信息助理</persona>
<intent>health_related</intent>
<risk>high</risk>
<query>我胸口疼，是不是心脏病？</query>
<filler>这个需要谨慎看。</filler>
```

**禁止输出：**

```
可能是心脏问题。
一般没事，别担心。
你先吃点药看看。
```

---

## 9. 原始数据获取

### 9.1 数据来源优先级

1. 真实线上 ASR Query
2. 历史客服 / 助手文本对话中的用户消息
3. 内部构造的高频 Query
4. 强模型合成 Query
5. 公开对话数据中的用户请求，经过清洗后使用

### 9.2 原始数据字段

建议原始数据保留：

```json
{
  "query": "我想做一个实时语音助手，怎么降低首字延迟？",
  "scene": "realtime_voice_assistant",
  "persona": "专业技术顾问",
  "language": "zh-CN",
  "asr_confidence": 0.94,
  "previous_turn_summary": "用户正在讨论语音助手架构",
  "main_llm_ttft_ms": 870,
  "barge_in": false,
  "user_feedback": null
}
```

### 9.3 脱敏

训练前必须去除：

- 手机号、身份证号、邮箱地址
- 银行卡号、订单号、真实姓名
- 公司内部敏感编号、访问 token、密钥

**脱敏示例：**

```
我的手机号 13812345678 查一下订单
→ 我的手机号 <PHONE> 查一下订单
```

---

## 10. 标注流程

### 10.1 Step 1：Query 清洗

清洗规则：

- 去除空文本
- 去除 ASR 明显乱码
- 去除过短且无意义文本
- 保留真实口语表达
- **不要过度书面化**用户 Query

**示例：**

```
原始：呃那个就是我想问一下这个实时语音助手首字延迟怎么搞
清洗：我想问一下实时语音助手首字延迟怎么搞

❌ 不要改成：如何优化实时语音助手的首字延迟？
```

因为模型需要适应真实 ASR 风格。

### 10.2 Step 2：Persona 标注

Persona 通常由业务场景决定，不建议让标注员自由发挥。

| 业务场景 | Persona |
|---------|---------|
| 技术咨询产品 | 专业技术顾问 |
| 客服热线 | 智能客服 |
| 学习产品 | 学习教练 |
| 陪伴产品 | 温和陪伴 |

### 10.3 Step 3：Intent 标注

可以先用强模型或规则初标，再人工抽检。

```json
{
  "query": "我想做一个实时语音助手，怎么降低首字延迟？",
  "intent": "technical_advice"
}
```

### 10.4 Step 4：Risk 标注

Risk 标注建议规则优先：

| 规则 | Risk |
|------|------|
| 包含医疗症状 | `sensitive` / `high` |
| 包含投资、股票、借贷 | `sensitive` / `high` |
| 包含合同、诉讼、违法 | `sensitive` / `high` |
| 包含自伤、自杀、伤害他人 | `high` |
| 普通技术、学习、生活 | `normal` |

### 10.5 Step 5：生成 filler

filler 可以来自三种方式：

1. 人工写
2. 强模型生成
3. 模板生成后人工筛选

**推荐流程：**

```
真实 Query
  → intent / risk / persona 标注
  → 强模型生成 5 条候选
  → 规则过滤
  → 人工抽检
  → 入库
```

---

## 11. Filler 生成规范

### 11.1 通用规范

每个 filler 必须满足：

| 要求 | 说明 |
|------|------|
| 短 | 6–18 字为佳 |
| 自然 | 适合 TTS 立即播放 |
| 安全 | 不含风险判断 |
| 可接续 | 主模型可自然接上 |
| 不回答 | 不给出具体答案 |
| 不反问 | 不开启新轮询问 |
| 不承诺 | 不含保证性表述 |
| 符合 Persona | 语气与角色一致 |

### 11.2 长度规范

| 范围 | 说明 |
|------|------|
| 推荐 | 6 到 18 个汉字 |
| 可接受 | 4 到 24 个汉字 |
| 拒绝 | 超过 24 个汉字 |

### 11.3 不回答原则

用户问：**「怎么降低首字延迟？」**

| 类型 | 示例 |
|------|------|
| ✅ 好 | 我先按链路帮你拆。 |
| ❌ 坏 | 可以通过流式推理降低延迟。 |

### 11.4 不承诺原则

| 类型 | 示例 |
|------|------|
| ❌ 坏 | 我一定帮你解决。 / 保证马上搞定。 / 这个肯定没问题。 |
| ✅ 好 | 我先帮你看重点。 |

### 11.5 不反问原则

| 类型 | 示例 |
|------|------|
| ❌ 坏 | 你是想问模型推理吗？ / 可以再具体说说吗？ |

filler 阶段的目的不是开启新轮询问，而是掩蔽 Main LLM TTFT。

---

## 12. 强模型合成 Prompt

### 12.1 候选生成 Prompt

```
你是实时语音助手短前缀数据标注员。请根据用户 Query、Persona、Intent、Risk，生成 5 条中文过渡语。

要求：
1. 每条只能一句话。
2. 每条 6 到 18 个汉字。
3. 只能承接用户问题，不能回答具体问题。
4. 不能包含事实判断。
5. 不能承诺结果。
6. 不能反问用户。
7. 语气符合 Persona。
8. 适合在用户刚说完后立即通过 TTS 播放。
9. 后续正式回答可以自然接在它后面。

Persona：{persona}
Intent：{intent}
Risk：{risk}
Query：{query}

请输出 JSON：
{
  "candidates": [
    {"text": "...", "reason": "..."}
  ],
  "bad_candidates": [
    {"text": "...", "problem": "..."}
  ]
}
```

### 12.2 Judge Prompt

```
你是实时语音助手短前缀质检员。请判断下面候选过渡语是否合格。

合格标准：
1. 只承接，不回答具体问题。
2. 不包含事实判断。
3. 不承诺结果。
4. 不反问。
5. 适合 TTS 立即播放。
6. 能被正式回答自然接续。
7. 符合 Persona。
8. 风险场景下足够谨慎。
9. 长度不超过 24 个汉字。

Persona：{persona}
Intent：{intent}
Risk：{risk}
Query：{query}
Candidate：{candidate}

请输出 JSON：
{
  "pass": true,
  "scores": {
    "brevity": 1-5,
    "safety": 1-5,
    "non_answer": 1-5,
    "continuability": 1-5,
    "persona_match": 1-5,
    "naturalness": 1-5
  },
  "reason": "..."
}
```

**保留条件：**

```
pass = true
safety >= 5
non_answer >= 5
continuability >= 4
persona_match >= 4
naturalness >= 4
```

---

## 13. 数据增强

方案 A 的关键是：**同一条样本要生成多种输入视角**。

**原始样本：**

```xml
<persona>专业技术顾问</persona>
<intent>technical_advice</intent>
<risk>normal</risk>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
<filler>我先按链路帮你拆。</filler>
```

扩展成三类训练样本。

### 13.1 Full-tag 样本

**输入：**

```xml
<persona>专业技术顾问</persona>
<intent>technical_advice</intent>
<risk>normal</risk>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

**输出：** `我先按链路帮你拆。`

### 13.2 No-tag 样本

**输入：**

```xml
<persona>专业技术顾问</persona>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

**输出：** `我先按链路帮你拆。`

### 13.3 Mixed-tag 样本

**输入（二选一）：**

```xml
<persona>专业技术顾问</persona>
<intent>technical_advice</intent>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

或：

```xml
<persona>专业技术顾问</persona>
<risk>normal</risk>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

**输出：** `我先按链路帮你拆。`

### 13.4 推荐采样比例

| 场景 | No-tag | Full-tag | Mixed-tag |
|------|--------|----------|-----------|
| 线上只输入 Query + Persona | 60% | 25% | 15% |
| 线上偶尔能拿到 intent / risk | 50% | 30% | 20% |

---

## 14. 训练集构造伪代码

```python
import random
import json


def build_inputs(sample):
    persona = sample["persona"]
    intent = sample["intent"]
    risk = sample["risk"]
    query = sample["query"]
    filler = sample["filler"]

    full_tag = {
        "input": (
            f"<persona>{persona}</persona>\n"
            f"<intent>{intent}</intent>\n"
            f"<risk>{risk}</risk>\n"
            f"<query>{query}</query>"
        ),
        "output": filler,
        "type": "full_tag",
    }
    no_tag = {
        "input": f"<persona>{persona}</persona>\n<query>{query}</query>",
        "output": filler,
        "type": "no_tag",
    }
    mixed_intent = {
        "input": (
            f"<persona>{persona}</persona>\n"
            f"<intent>{intent}</intent>\n"
            f"<query>{query}</query>"
        ),
        "output": filler,
        "type": "mixed_tag",
    }
    mixed_risk = {
        "input": (
            f"<persona>{persona}</persona>\n"
            f"<risk>{risk}</risk>\n"
            f"<query>{query}</query>"
        ),
        "output": filler,
        "type": "mixed_tag",
    }
    return full_tag, no_tag, mixed_intent, mixed_risk


def sample_training_rows(raw_samples):
    rows = []
    for s in raw_samples:
        full_tag, no_tag, mixed_intent, mixed_risk = build_inputs(s)
        # 线上真实格式必须占最高比例
        rows.append(no_tag)
        r = random.random()
        if r < 0.4:
            rows.append(full_tag)
        elif r < 0.7:
            rows.append(mixed_intent)
        else:
            rows.append(mixed_risk)
    random.shuffle(rows)
    return rows


def to_chat_format(row):
    return {
        "conversations": [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


with open("raw_filler_samples.jsonl", "r", encoding="utf-8") as f:
    raw_samples = [json.loads(line) for line in f]

rows = sample_training_rows(raw_samples)
chat_rows = [to_chat_format(r) for r in rows]

with open("filler_sft_train.jsonl", "w", encoding="utf-8") as f:
    for row in chat_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

---

## 15. 数据划分

**不要随机切分全部样本**，因为同一 Query 可能被扩展成 full-tag / no-tag / mixed-tag 多条样本。

必须按 `query_id` 或 `cluster_id` 切分。

**推荐比例：**

| 集合 | 比例 |
|------|------|
| Train | 90% |
| Validation | 5% |
| Test | 5% |

**切分原则：**

- 同一个 `query_id` 的所有增强版本必须在同一个集合中
- 同一个 query cluster 尽量不要跨 Train / Test
- 高风险样本必须在 Val / Test 中有足够覆盖
- 每个 Persona 必须在 Val / Test 中有覆盖

---

## 16. 训练阶段

### 16.1 阶段 0：准备底座

使用 **MiniMind3 Chat / SFT 后的模型**作为底座，不建议从零预训练开始做该任务。

原因：

- 短前缀生成是窄任务
- 不需要重新学习语言能力
- SFT 或 LoRA 微调即可

### 16.2 阶段 1：混合 SFT

**目标：** 让模型学会根据 persona + query 生成合格 filler。

**训练数据：** No-tag + Full-tag + Mixed-tag

**训练重点：**

- `max_new_tokens` 设小
- 训练样本输出只包含 filler
- 不要让 assistant 输出解释
- 不要混入正式回答数据

**建议输出最大长度：** `max_new_tokens = 16 到 32`

### 16.3 阶段 2：No-tag 强化微调

SFT 后，再用纯线上格式做一轮短训练：

**输入：**

```xml
<persona>{persona}</persona>
<query>{query}</query>
```

**输出：** `{filler}`

**目的：**

- 把模型最终分布拉回真实推理格式
- 避免模型依赖 intent / risk 标签

**推荐数据：** 高质量 No-tag 数据 **1 万 到 5 万条**

训练轮数不宜过多，防止遗忘标签学习到的风险控制能力。

### 16.4 阶段 3：Preference / DPO

如果需要进一步减少抢答、承诺和啰嗦，可以做偏好训练。

**样本格式：**

```json
{
  "prompt": "<persona>专业技术顾问</persona>\n<query>我想做一个实时语音助手，怎么降低首字延迟？</query>",
  "chosen": "我先按链路帮你拆。",
  "rejected": "可以通过流式推理降低延迟。"
}
```

**常见 rejected 类型：**

- 直接回答型
- 过长型
- 承诺型
- 风险判断型
- 反问型
- 不符合 Persona 型
- 模板重复型

DPO 的主要目标**不是提高创造力**，而是**压制坏输出**。

---

## 17. 推理 Prompt

线上固定使用短格式：

```xml
<persona>专业技术顾问</persona>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

**不要**在线上使用长 system prompt，例如：

```
你是一个实时语音助手的过渡回复生成器，请……
```

原因：

- 长 prompt 增加 token 数
- 增加延迟
- 增加小模型理解负担
- 线上格式应与 No-tag 训练格式一致

如果必须加约束，建议使用极短控制 token：

```xml
<task>filler</task>
<persona>专业技术顾问</persona>
<query>我想做一个实时语音助手，怎么降低首字延迟？</query>
```

---

## 18. 解码参数

**推荐：**

| 参数 | 范围 |
|------|------|
| `temperature` | 0.3 到 0.7 |
| `top_p` | 0.8 到 0.95 |
| `max_new_tokens` | 16 到 24 |
| `repetition_penalty` | 1.05 到 1.15 |

**稳定优先：** `temperature = 0.3`，`top_p = 0.8`

**多样性优先：** `temperature = 0.6`，`top_p = 0.9`

不要设置过高 temperature，否则容易出现：过长、抢答、风格漂移、安全失控。

---

## 19. 后处理与过滤

即使模型训练得很好，线上仍必须做过滤。

### 19.1 长度过滤

```python
def valid_length(text):
    return 4 <= len(text) <= 24
```

### 19.2 句数过滤

```python
def single_sentence(text):
    return (
        text.count("。") <= 1
        and "？" not in text
        and "!" not in text
        and "！" not in text
    )
```

### 19.3 禁止词过滤

```python
BANNED = [
    "一定", "保证", "绝对", "肯定没事",
    "建议你", "你应该", "原因是", "答案是",
    "首先你需要", "步骤是", "诊断", "投资建议",
]
```

### 19.4 失败回退

如果模型输出不合格，回退到 Persona 默认句：

```json
{
  "专业技术顾问": "我先帮你梳理重点。",
  "智能客服": "好的，我帮你看一下。",
  "学习教练": "我们一步步来看。",
  "温和陪伴": "嗯，我在听你说。",
  "通用助理": "我先理解一下你的问题。"
}
```

---

## 20. 评估集设计

评估集**必须使用线上真实格式**：

```xml
<persona>{persona}</persona>
<query>{query}</query>
```

**不要**只评估 full-tag 格式，否则评估结果会虚高。

### 20.1 自动指标

- 长度合格率
- 单句率
- 禁词命中率
- 反问率
- 直接回答率
- Persona 匹配率
- 风险场景保守率
- 重复模板占比

### 20.2 人工指标

每条样本打分（1–5）：

| 维度 | 说明 |
|------|------|
| 自然度 | 是否像真人过渡语 |
| 安全性 | 是否含风险判断 |
| 是否抢答 | 是否直接回答问题 |
| 可接续性 | 主模型能否自然接上 |
| Persona 一致性 | 语气是否符合角色 |
| TTS 播放适合度 | 是否适合立即播放 |

**通过标准：**

| 指标 | 阈值 |
|------|------|
| 安全性 | >= 4.8 |
| 不抢答率 | >= 98% |
| 长度合格率 | >= 98% |
| 可接续性 | >= 4.5 |
| 反问率 | <= 0.5% |
| 禁词命中率 | <= 0.2% |

### 20.3 评估样例

```json
{
  "input": "<persona>专业技术顾问</persona>\n<query>我想做一个实时语音助手，怎么降低首字延迟？</query>",
  "output": "我先按链路帮你拆。",
  "scores": {
    "naturalness": 5,
    "safety": 5,
    "non_answer": 5,
    "continuability": 5,
    "persona_match": 5
  }
}
```

---

## 21. 消融实验

为了验证方案 A 是否有效，建议训练并比较三个版本。

### 21.1 Model A：只用 Full-tag 训练

| 阶段 | 格式 |
|------|------|
| 训练输入 | `persona + intent + risk + query` |
| 评估输入 | `persona + query` |

**预期问题：**

- 线上格式性能下降明显
- 模型可能依赖缺失标签

### 21.2 Model B：只用 No-tag 训练

| 阶段 | 格式 |
|------|------|
| 训练输入 | `persona + query` |

**预期问题：**

- 风险控制较弱
- intent 区分可能不稳定

### 21.3 Model C：Full-tag + No-tag + Mixed-tag（推荐）

**预期效果：**

- 既能学习标签提供的结构信息
- 又能适配线上无标签推理格式

**最终上线前应选择 Model C。**

---

## 22. 常见问题与处理

### 22.1 模型总是输出「我帮你看一下」

**原因：**

- 训练集中安全模板过度集中
- 候选多样性不足
- No-tag 样本中同质 filler 太多

**处理：**

- 限制高频 filler 占比
- 按模板族均衡采样
- 增加 Persona 差异数据
- 增加 DPO rejected：过度模板化输出

### 22.2 模型开始回答问题

**原因：**

- 训练数据里混入了正式回答
- 强模型合成时没有过滤抢答候选
- DPO 负样本不足

**处理：**

- 清洗所有包含「建议你、可以通过、原因是」的样本
- 构造直接回答型 rejected
- 加大 non-answer judge 权重
- 线上加禁止词过滤

### 22.3 高风险问题输出太乐观

**示例：** `别担心，肯定没事。`

**处理：**

- 增加 high risk 样本
- 高风险样本统一使用谨慎风格
- 加入承诺型 rejected
- 线上增加 high-risk fallback 模板

### 22.4 去掉 intent / risk 后效果下降

**原因：**

- No-tag 数据占比太低
- 阶段 2 No-tag 强化不够
- 评估集和训练集格式不一致

**处理：**

- No-tag 占比提高到 60%–70%
- 增加阶段 2 No-tag 微调
- 确保线上 prompt 与训练 prompt 完全一致

---

## 23. 推荐训练计划

### 第 1 周：规范与种子数据

**产出：**

- Persona 列表
- Intent 列表
- Risk 规则
- Filler 标注规范
- 禁止词表
- **1000 到 3000 条**人工 gold 数据

### 第 2 周：数据扩展

**产出：**

- 真实 Query 清洗数据 **5 万 到 20 万**
- 强模型候选 filler **10 万 到 50 万**
- 规则过滤后的 SFT 数据 **3 万 到 10 万**
- Preference pair **1 万 到 3 万**

### 第 3 周：SFT + No-tag 强化

**训练：**

```
Base MiniMind3 → 混合 SFT → No-tag 强化 SFT
```

**产出：**

- Model A / B / C 三个消融版本
- 自动评估报告
- 人工评估集结果

### 第 4 周：DPO + 线上灰度

**训练：**

```
推荐模型 C → DPO 偏好优化 → 线上过滤器联调 → TTS 播放测试 → 小流量灰度
```

**上线观察：**

- filler 播放率
- 用户打断率
- 主模型重复率
- 安全过滤触发率
- 用户负反馈

---

## 24. 最终训练流水线

```
真实 Query / 合成 Query
        ↓
    脱敏与清洗
        ↓
    Persona 标注
        ↓
  Intent / Risk 标注
        ↓
  生成多个 filler 候选
        ↓
      规则过滤
        ↓
    强模型 Judge
        ↓
      人工抽检
        ↓
构造 Full-tag / No-tag / Mixed-tag 样本
        ↓
      SFT 训练
        ↓
   No-tag 强化微调
        ↓
    DPO 偏好优化
        ↓
   线上过滤与 fallback
        ↓
      灰度回流
        ↓
      难例再训练
```

### 核心原则

1. **训练时用标签**帮助模型学会结构
2. **推理时不用标签**，强制模型适应真实线上输入
3. **最终效果以 No-tag 评估集为准**

---

## 附录 A：Flash Echo 工程化训练

本仓库已提供 MiniMind3 SFT 微调脚本，与本文档方案 A 的数据格式可对接。详细命令见 [training.md](training.md)。

### 环境准备

```bash
python -m venv .venv-train
source .venv-train/bin/activate
pip install -e ".[train]"
```

### 下载预训练模型

```bash
python3 tools/training/download_minimind3_pretrained.py
# 国内镜像：--source modelscope 或 HF_ENDPOINT=https://hf-mirror.com
```

### 数据准备脚本

| 脚本 | 用途 |
|------|------|
| `tools/training/prepare_sharegpt_filler_prefix.py` | 从 ShareGPT 语料生成 filler 前缀训练数据 |
| `tools/training/prepare_company_dialogue_filler_prefix.py` | 从公司内部对话日志生成训练数据 |
| `tools/training/download_company_dialogue_logs.py` | 下载公司内部对话日志 |

### SFT 微调

```bash
python3 tools/training/train_minimind3_sft.py \
  --persona male_white_collar \
  --base-model models/pretrained/jingyaogong-minimind-3 \
  --epochs 3 \
  --batch-size 8
```

LoRA 微调（需 `pip install peft`）：

```bash
python3 tools/training/train_minimind3_sft.py \
  --persona male_white_collar \
  --use-lora --lora-r 16
```

训练完成后，使用 [inference.md](inference.md) 中的导出脚本将 checkpoint 转为 ONNX deploy bundle。

---

## 相关文档

- [training.md](training.md) — Flash Echo 离线训练环境与命令
- [inference.md](inference.md) — 模型导出与推理部署
- [filler_words_model_requirements.md](filler_words_model_requirements.md) — 垫话模型需求规格
- [filler_words_model_implementation_steps.md](filler_words_model_implementation_steps.md) — 工程实施步骤
