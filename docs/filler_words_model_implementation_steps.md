# 垫话/过渡回复模型实施步骤

本文档基于 `docs/filler_words_model_requirements.md`，将“极轻量、高并发垫话/过渡回复模型”的需求拆解为可执行的工程实施步骤。目标是在实时语音交互链路中，先用静态规则快速处理高频完整短回复，再按 Persona 路由到对应 MiniMind3 短前缀模型生成可续写垫话，并通过 OpenAI 兼容接口声明下游是否需要继续调用 Qwen-Plus。

---

## 1. 实施目标与边界确认

### 1.1 实施目标

1. 构建一个 OpenAI 兼容的 `POST /v1/chat/completions` 服务或模块，输出“当前要立即播放文本”。
2. 用 Static Reply Gate 处理寒暄、结束、感谢、确认、取消等无需主模型续写的高频 Query。
3. 对未命中静态规则的 Query，根据 `metadata.persona_tag` 路由到 Persona 专属 MiniMind3 SFT/LoRA 模型，生成一句短、自然、可续写的回复前缀。
4. 对 MiniMind3 输出进行长度、标点、禁答、事实污染、操作承诺和续写可行性校验。
5. 在 MiniMind3 超时、不可用或输出不合法时，返回版本化的兜底短前缀。
6. 在 `filler_reply` 中输出 `route`、`reply_kind`、`continue_with_main_llm`、`rule_id`、`model_version`、`fallback_reason` 等元信息，支持灰度、回滚、监控和离线分析。

### 1.2 项目边界

本项目只负责：

1. 接收当前轮用户 Query、Persona 和可选链路追踪字段。
2. 判断是否命中静态短回复规则。
3. 未命中规则时完成 Persona 专属 MiniMind3 模型路由、短前缀生成、输出校验和兜底。
4. 通过 `choices[0].message.content` 返回可立即播放的文本，并通过响应扩展元信息标记是否需要主模型续写。
5. 为规则、模型、兜底和缓存行为记录可观测元信息。

本项目不负责：

1. ASR、VAD、TTS、主 LLM 调用。
2. 音频拼接、播放策略和多路并发编排。
3. Qwen-Plus 的具体 Prompt 编排和去重实现，但必须通过接口约定要求下游感知已播放前缀。

---

## 2. 阶段一：接口契约与核心枚举固化

### 2.1 固化输入字段

服务入口采用 OpenAI Chat Completions 兼容格式：

```text
POST /v1/chat/completions
```

请求字段必须保持精简，避免增加实时链路负担：

```json
{
  "model": "filler-reply-minimind3",
  "messages": [
    {
      "role": "user",
      "content": "你觉得人工智能未来会毁灭人类吗？"
    }
  ],
  "stream": false,
  "metadata": {
    "persona_tag": "female_receptionist",
    "locale": "zh-CN",
    "request_id": "req_001"
  }
}
```

字段要求：

1. `model`：必填，OpenAI 兼容模型名。建议使用服务级模型名，例如 `filler-reply-minimind3`；真实 Persona 专属 MiniMind3 模型由服务内部根据 `metadata.persona_tag` 路由。
2. `messages`：必填，兼容 OpenAI Chat Completions。服务取最后一条 `role=user` 的 `content` 作为当前轮 Query；v1 阶段默认不拼接历史上下文。
3. `stream`：可选，初版建议仅支持 `false`。垫话文本很短，不需要流式输出；当客户端传入 `stream=true` 且服务尚未实现流式兼容时，应返回 OpenAI 兼容错误响应（建议 HTTP 400 或 501），不得静默忽略。
4. `metadata.persona_tag`：必填，用于静态短回复候选选择和 MiniMind3 专属模型路由，不作为 MiniMind3 的文本输入参数。
5. `metadata.locale`：可选，默认 `zh-CN`，用于后续多语言扩展。
6. `metadata.request_id`：可选，用于链路追踪和日志关联。
7. Query 长度：必须定义最大输入长度（按字符数或 Token 数），超过上限时按产品策略返回错误或截断到安全长度；截断行为需要记录日志，且不得影响静态规则的保守判断。

### 2.2 固化输出字段

服务响应保持 OpenAI Chat Completions 外壳。当前应立即播放的文本放在 `choices[0].message.content`，业务编排元信息放在顶层扩展字段 `filler_reply`：

```json
{
  "id": "chatcmpl-req_001",
  "object": "chat.completion",
  "created": 1779979680,
  "model": "filler-reply-minimind3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "这个问题可以这样看，"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 7,
    "total_tokens": 31
  },
  "filler_reply": {
    "reply_kind": "filler_prefix",
    "continue_with_main_llm": true,
    "route": "minimind3_filler",
    "rule_id": null,
    "model_version": "minimind3-filler-female-receptionist-v1.0.0",
    "fallback_reason": null,
    "request_id": "req_001"
  }
}
```

字段要求：

1. `choices[0].message.content`：当前要立刻播放的文本。静态规则命中时是完整短回复；MiniMind3 或兜底路径命中时是可续写短前缀。
2. `choices[0].finish_reason`：固定为 `stop`。
3. `usage`：保持 OpenAI 兼容字段；MiniMind3 路径可返回实际或估算 token，静态规则路径建议统一返回 `0`，或按服务统计口径估算，但同一部署内必须保持一致。
4. `filler_reply.reply_kind`：取值为 `complete_static_reply` 或 `filler_prefix`。
5. `filler_reply.continue_with_main_llm`：静态完整短回复为 `false`，短前缀路径为 `true`。
6. `filler_reply.route`：取值为 `static_reply`、`minimind3_filler` 或 `fallback_filler`。
7. `filler_reply.rule_id`：静态规则命中时返回规则 ID，其他路径为空。
8. `filler_reply.model_version`：MiniMind3 模型版本；静态规则命中时可为空。
9. `filler_reply.fallback_reason`：兜底路径触发原因，例如 `timeout`、`invalid_length`、`unsafe_output`、`model_unavailable`。
10. `filler_reply.request_id`：透传 `metadata.request_id`。

### 2.3 固化核心枚举

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| `route` | `static_reply` | 命中静态规则，返回完整短回复 |
| `route` | `minimind3_filler` | 未命中静态规则，MiniMind3 生成短前缀 |
| `route` | `fallback_filler` | MiniMind3 超时、不可用或输出不合法，使用兜底短前缀 |
| `reply_kind` | `complete_static_reply` | 完整静态短回复，不需要主模型续写 |
| `reply_kind` | `filler_prefix` | 可续写垫话前缀，需要主模型从后面继续回答 |
| `fallback_reason` | `timeout` | MiniMind3 推理超时 |
| `fallback_reason` | `model_unavailable` | Persona 专属模型不可用或路由失败 |
| `fallback_reason` | `invalid_length` | 输出过短或过长 |
| `fallback_reason` | `invalid_boundary` | 输出没有以 `，`、`。` 或 `！` 结尾 |
| `fallback_reason` | `direct_answer` | 输出直接回答了用户问题 |
| `fallback_reason` | `unsafe_output` | 输出存在安全、事实污染或操作承诺风险 |

交付物：

1. OpenAI 兼容 API 请求与响应 Schema。
2. 核心枚举定义文件。
3. 下游集成契约说明。
4. 示例请求和三类路径的示例响应。

---

## 3. 阶段二：静态短回复规则库建设

### 3.1 明确规则覆盖范围

Static Reply Gate 只覆盖高置信、高频、低歧义、无需 Qwen-Plus 续写的 Query：

1. 问候：如“你好”“您好”“哈喽”“嗨”。
2. 结束：如“再见”“拜拜”“下次聊”。
3. 感谢：如“谢谢”“多谢”“辛苦了”。
4. 确认：如“好的”“好”“嗯”“行”“可以”。
5. 取消/否定：如“不用了”“算了”“没事了”。

任务请求、开放问题、事实问题、推理问题、情绪表达、检索请求、指代不明请求不得被静态规则拦截，默认进入 MiniMind3 短前缀路径。

### 3.2 定义规则配置结构

静态规则建议使用 JSON、YAML 或数据库配置，初版可先使用版本化配置文件：

```json
{
  "version": "static-rules-v1",
  "locale": "zh-CN",
  "rules": [
    {
      "rule_id": "thanks.reply.001",
      "category": "thanks",
      "match": {
        "type": "exact",
        "patterns": ["谢谢", "多谢", "辛苦了"]
      },
      "negative_examples": ["谢谢你帮我分析一下这个合同"],
      "replies": {
        "female_receptionist": ["不客气！", "应该的！"],
        "male_white_collar": ["不客气。"],
        "default": ["不客气！"]
      },
      "enabled": true
    }
  ]
}
```

字段要求：

1. `rule_id` 全局唯一，并能反映规则类别。
2. `match.type` 取值为 `exact`、`exact_with_trailing_punctuation`、`prefix`、`regex` 之一（见 `StaticRuleMatchType`）：
   - `exact`：归一化后与 `patterns` 中任一项完全一致，可用于单个标准表达或多个等价表达。
   - `exact_with_trailing_punctuation`：在 `exact` 基础上，允许 Query 句尾带非问号类语气/停顿标点（如 `。`、`！`、`，`、`~`），但不去除 `？` 或 `?`。
   - `prefix`：归一化后以 `patterns` 中某一项为前缀。
   - `regex`：归一化后全文匹配 `patterns` 中的正则（`fullmatch`）；仅用于少量保守规则。
   - 不再支持历史值 `alias`、`exact_or_alias`；需要使用 `exact`。
3. `negative_examples` 必须覆盖容易误拦截的真实问题。
4. `replies` 按 Persona 配置候选完整短回复，并提供 `default` 兜底。
5. 每条规则至少配置 3 到 10 条候选短回复，候选不足时可先固定返回，但需要在验收中标记。

`negative_examples` 与 hard cases 的边界：

1. `negative_examples` 是规则级反例，随具体规则配置维护，用于说明该规则不应命中的相似 Query。
2. hard cases 是跨规则、跨模型版本的动态回归集，来自线上误拦截、不合法输出、兜底高频 Query 和人工复核样本。
3. 规则上线前应同时检查本规则的 `negative_examples` 和全局 hard cases，避免只通过局部反例。

### 3.3 实现 Query 归一化

规则匹配前只做低成本文本归一化：

1. 去除首尾空白。
2. 统一全半角。
3. 规整常见中英文标点。
4. 合并重复空格。
5. 可选去除句尾语气标点，但必须保留足够信息避免误判。

归一化逻辑需要版本化，并在日志中记录归一化版本，便于规则命中异常回溯。

### 3.4 规则审核与回归

规则上线前必须完成：

1. 每条规则有命中样例、反例、业务动机和回滚信息。
2. 使用线上高频 Query 回放，统计静态规则误拦截率。
3. 误拦截率建议 `< 1%`。
4. 边界样本进入 hard cases 集，并作为后续规则和模型版本的固定回归集。
5. 规则发布支持热更新或灰度，并记录规则版本。

交付物：

1. 静态规则配置文件。
2. Query 归一化实现与版本说明。
3. 规则审核清单。
4. 静态规则回归测试集。
5. 规则命中率和误拦截评估报告。

---

## 4. 阶段三：MiniMind3 SFT 数据集建设

### 4.1 数据结构

MiniMind3 不再预测功能类别，也不接收 Persona 文本参数。每个 Persona 单独维护 SFT 数据集，单条样本只包含用户 Query 和该 Persona 风格下的短前缀：

```json
{"query":"我最近真的很焦虑，不知道怎么办","filler_prefix":"我能理解你现在的压力，","source":"emotion_dialog","label_method":"qwen_plus_reviewed"}
{"query":"你怎么看 AI 对教育行业的影响？","filler_prefix":"这个问题可以这样看，","source":"open_question","label_method":"qwen_plus_reviewed"}
```

数据集目录建议：

```text
data/filler_prefix/
  female_receptionist/
    train.jsonl
    val.jsonl
    test.jsonl
    hard_cases.jsonl
  male_white_collar/
  grandpa/
  young_girl/
```

### 4.2 语料来源

优先收集以下 Query：

1. 线上真实 Query，需脱敏、去重、聚类后进入候选集。
2. 常见 QA、闲聊、任务型请求、开放观点、情绪表达、检索请求和指代不明请求。
3. 静态规则未覆盖但高频出现的边界 Query。
4. MiniMind3 历史输出不合法或体验较差的 hard cases。

初版建议规模：

1. 每个 Persona 最小可用训练集 5,000 到 10,000 条。
2. 初版上线候选训练集每个 Persona 30,000 到 100,000 条。
3. hard cases 集固定版本化保存，每次 SFT 或 LoRA 迭代必须回归。

### 4.3 使用强模型生成短前缀

可使用 Qwen-32B、Qwen-Plus 或更强模型批量生成高质量短前缀样本。以下 Prompt 仅用于离线数据合成；线上 MiniMind3 推理阶段仍只接收 Query 文本，不拼接 Persona 描述。生成 Prompt 必须强调：

1. 输入只有用户 Query。
2. 输出一句适合当前 Persona 的中文短垫话前缀。
3. 长度建议 5 到 18 个中文字符，特殊情况下不超过 25 个中文字符。
4. 必须以 `，`、`。` 或 `！` 结尾。
5. 不得直接回答问题。
6. 不得编造具体时间、地点、人物、数据或外部事实。
7. 不得承诺未执行的检索、预订、查询或操作。
8. 必须能被 Qwen-Plus 作为已播放前缀自然续写。

初版生成 Prompt 示例：

```text
你是实时语音交互系统的数据合成助手。请根据用户 Query 生成一句中文短垫话前缀，用于主模型回答前立即播放。

Persona:
female_receptionist，语气自然、礼貌、温和，不夸张。

要求：
1. 只输出一句短前缀，不要解释。
2. 长度优先控制在 5 到 18 个中文字符，最长不超过 25 个中文字符。
3. 必须以中文逗号、句号或感叹号结尾。
4. 不要直接回答用户问题。
5. 不要编造事实、数据、时间、地点或人物。
6. 不要说“我查到了”“我已经帮你处理好了”等未发生的操作。
7. 输出必须能让主模型从后面自然继续回答。

用户 Query:
{query}
```

### 4.4 数据校验与人工复核

样本入库前执行自动校验：

1. 长度是否在允许范围内。
2. 是否以合法标点结尾。
3. 是否包含明显直接答案。
4. 是否包含事实污染或操作承诺。
5. 是否出现重复、空泛、无法续写或 Persona 风格错配。

需要人工复核的样本：

1. 自动校验失败样本。
2. 强模型多次生成结果差异较大的样本。
3. 高频线上 Query。
4. 情绪、安全、医疗、法律、金融等高风险 Query。
5. Persona 风格容易夸张或冒犯的样本。

交付物：

1. Persona 分组 SFT 数据集。
2. 数据生成 Prompt 与批处理脚本。
3. 自动校验脚本。
4. 人工复核池与复核结果。
5. hard cases 数据集。

---

## 5. 阶段四：MiniMind3 模型训练、评估与导出

### 5.1 训练策略

每个 Persona 训练或微调一个独立 MiniMind3 短前缀模型：

1. `female_receptionist` 对应 `minimind3-filler-female-receptionist-v1`。
2. `male_white_collar` 对应 `minimind3-filler-male-white-collar-v1`。
3. `grandpa` 对应 `minimind3-filler-grandpa-v1`。
4. `young_girl` 对应 `minimind3-filler-young-girl-v1`。

示例中的 `v1` 是简写，正式 `model_version` 应采用 5.5 节的三段式版本号，例如 `minimind3-filler-female-receptionist-v1.0.0`。

训练输入格式建议保持简单：

```text
用户：{query}
请生成一句可续写的短垫话前缀：
```

训练输出为对应 Persona 的 `filler_prefix`。不要把 `persona_tag` 拼接进单次模型输入，Persona 由服务路由到不同模型来控制。

### 5.2 SFT 与 LoRA 配置

训练时重点控制短输出能力，而不是开放回答能力：

1. 使用 SFT 或 LoRA 微调 MiniMind3。
2. 训练样本输出只保留短前缀，不加入完整答案。
3. 控制最大输出 token，避免模型学习长回答。
4. 保持各 Query 类型分布均衡，尤其是开放问题、任务请求、情绪表达和指代不明请求。
5. hard cases 在训练集和评估集中单独分桶统计。
6. 每个 Persona 独立评估，不用一个模型混合多个 Persona 风格。

### 5.3 离线自动评估

模型候选版本需要至少满足：

1. 合法输出率 `>= 98%`，覆盖长度、标点、安全、禁答和续写校验。
2. 禁答违规率 `< 1%`。
3. 事实污染率 `< 0.5%`。
4. hard cases 集无明显退化。
5. 输出重复率在可接受范围内，并按 Persona 分桶统计。

自动评估需要记录原始 Query、Persona、模型输出、失败原因、模型版本和数据版本。

### 5.4 人工体验评估

人工评估维度：

1. 前缀是否自然、口语化、适合 TTS。
2. 是否符合 Persona 风格，但不过度表演。
3. 是否没有直接回答问题。
4. 是否能被 Qwen-Plus 从后面自然续写。
5. 是否存在重复、突兀、冒犯或语气割裂。

上线候选建议达到：

1. 前缀自然度平均分 `>= 4.2 / 5`。
2. 续写连贯性通过率 `>= 95%`。
3. Persona 风格匹配度达到人工验收标准。

### 5.5 导出与版本管理

模型包需要包含：

1. MiniMind3 权重或 LoRA adapter。
2. Tokenizer 文件。
3. 推理配置，包括最大输出 token、temperature、top_p、stop 规则和超时预算。
4. `model_version`。
5. 训练数据版本和评估报告。
6. Persona 路由元信息。

版本命名建议：

```text
minimind3-filler-{persona-slug}-v{major}.{minor}.{patch}
```

交付物：

1. Persona 专属训练配置。
2. SFT/LoRA 训练脚本。
3. 模型评估报告。
4. 可部署模型包。
5. 模型版本与数据版本映射表。

---

## 6. 阶段五：推理服务实现

### 6.1 实现三段式主流程

服务主流程：

1. 校验 OpenAI 兼容请求体，确保 `model`、`messages` 和 `metadata.persona_tag` 存在。
2. 从最后一条 `role=user` 的 `content` 提取当前轮 Query，校验 Query 长度，并做轻量归一化。
3. 查询 Static Reply Gate。
4. 若命中静态规则，根据 Persona 从候选短回复中选择文本，返回 OpenAI Chat Completions 响应，并在 `filler_reply` 中写入 `route=static_reply`、`reply_kind=complete_static_reply`、`continue_with_main_llm=false`。
5. 若未命中静态规则，根据 `metadata.persona_tag` 查找 Persona 专属 MiniMind3 模型。
6. 若模型不可用或路由失败，进入兜底路径。
7. 对 Query 做 Prompt 注入防护后调用 MiniMind3 生成短前缀，包括长度截断、控制字符过滤和明确用户输入边界。
8. 对输出执行后处理校验。
9. 校验通过时返回 OpenAI Chat Completions 响应，并在 `filler_reply` 中写入 `route=minimind3_filler`、`reply_kind=filler_prefix`、`continue_with_main_llm=true`。
10. 校验失败或超时时返回兜底短前缀，并记录 `fallback_reason`。
11. 记录结构化日志和指标。

### 6.2 实现 Persona 模型路由

路由配置建议独立版本化：

```json
{
  "version": "model-router-v1",
  "routes": {
    "female_receptionist": {
      "model_version": "minimind3-filler-female-receptionist-v1.0.0",
      "enabled": true
    },
    "male_white_collar": {
      "model_version": "minimind3-filler-male-white-collar-v1.0.0",
      "enabled": true
    }
  }
}
```

路由原则：

1. `metadata.persona_tag` 必须路由到对应 Persona 专属模型。
2. 不允许在路由失败时静默使用其他 Persona 模型，避免风格错配。
3. 路由失败应进入 `fallback_filler`，并记录 `fallback_reason=model_unavailable`。
4. 路由配置支持灰度、禁用和快速回滚。
5. 未知 `persona_tag`（请求值存在但不在路由表中）按路由失败处理，进入兜底路径并记录告警，不得静默映射到 `default` 或其他 Persona 模型。

### 6.3 实现输出后处理校验

MiniMind3 输出播放前必须通过：

1. 长度校验：中文长度建议 5 到 18 字，最长不超过 25 字。
2. 边界校验：必须以 `，`、`。` 或 `！` 结尾。
3. 禁答校验：不得直接回答用户问题。
4. 事实校验：不得编造具体时间、地点、人物、数据或外部事实。
5. 操作承诺校验：不得承诺未执行的检索、预订、查询或处理。
6. 续写校验：文本必须能被 Qwen-Plus 从后面自然接上。
7. 安全校验：不得输出冒犯、歧视、露骨或其他不适合 TTS 播放的内容。

安全校验实现需纳入延迟预算。v1 阶段优先采用本地轻量规则、敏感词表或本地分类器；若接入外部内容审核服务，必须设置独立超时，并在延迟分解中单独统计安全校验耗时。

校验失败时不要把原始模型输出返回给下游，应立即使用兜底短前缀。MiniMind3 推理超时时，无论是否已产生部分 token，都必须丢弃部分输出并进入兜底路径，不对不完整输出做播放或校验。

### 6.4 实现兜底短前缀库

兜底库按 Locale 和 Persona 维护，内容必须短、通用、可续写：

```json
{
  "version": "fallback-prefix-v1",
  "zh-CN": {
    "female_receptionist": ["我想一下，", "我来帮你梳理一下，"],
    "male_white_collar": ["我想一下，", "这个问题可以这样看，"],
    "default": ["我想一下，", "我来帮你梳理一下，"]
  }
}
```

要求：

1. 兜底前缀必须通过与 MiniMind3 输出相同的长度、标点和续写校验。
2. 兜底路径必须返回 `route=fallback_filler`。
3. `fallback_reason` 必须明确记录。
4. 兜底库需要版本化，并支持灰度和回滚。

### 6.5 异常处理

建议策略：

1. `messages` 缺失、最后一条用户消息为空或无法提取 Query：返回 OpenAI 兼容错误响应，或按产品策略返回兜底短前缀；不应进入模型推理。
2. Query 超过最大长度：按配置返回 OpenAI 兼容错误响应或截断到安全长度；截断时必须记录原始长度、截断后长度和处理策略。
3. `stream=true` 但服务未实现流式兼容：返回 OpenAI 兼容错误响应（建议 HTTP 400 或 501）。
4. `metadata.persona_tag` 缺失：返回 OpenAI 兼容错误响应；若产品要求强可用，可使用默认 Persona 的兜底短前缀，并记录告警。
5. `metadata.persona_tag` 未知或未配置路由：进入 `fallback_filler`，记录 `fallback_reason=model_unavailable` 和告警。
6. 静态规则配置加载失败：禁用 Static Reply Gate，所有请求进入 MiniMind3 或兜底路径，并触发告警。
7. MiniMind3 超时：立即返回兜底短前缀，并丢弃任何部分输出。
8. 后处理校验异常：返回兜底短前缀，并记录 `fallback_reason=unsafe_output` 或具体失败原因。

交付物：

1. 推理服务代码。
2. Static Reply Gate 实现。
3. Persona 模型路由实现。
4. MiniMind3 推理适配层。
5. 输出校验与兜底模块。
6. API 文档和错误码定义。

---

## 7. 阶段六：缓存、部署与性能优化

### 7.1 静态规则与配置加载

部署要求：

1. 静态规则、兜底库、模型路由配置随服务启动加载到内存。
2. 配置文件带版本号，日志中记录实际命中的版本。
3. 支持热更新或灰度发布。
4. 配置更新失败时保留上一可用版本，并触发告警。

### 7.2 MiniMind3 结果缓存

对未命中静态规则但线上高频重复的 Query，可建立内存级 KV 缓存：

1. Key：`normalized_query + persona_tag + model_version + static_rules_version + normalizer_version`。
2. Value：`text + validation_result + generated_at + hit_count`。
3. 缓存命中时可以跳过模型推理，但不能跳过输出有效性和版本检查。
4. 缓存值需要记录命中次数，便于发现高频不自然前缀并进入人工复核。
5. 静态规则或归一化逻辑热更新时，需要按版本隔离缓存或淘汰受影响缓存，避免新规则下本应命中的 Query 继续复用旧 MiniMind3 结果。
6. 缓存容量、TTL 和淘汰策略应按实例内存预算配置。

静态规则路径不需要模型结果缓存，但需要统计规则命中率和规则级延迟。

### 7.3 推理引擎与部署形态

MiniMind3 优先本地部署，可按工程栈选择：

1. llama.cpp。
2. vLLM。
3. ONNX Runtime。
4. TensorRT-LLM。
5. 其他轻量推理引擎。

服务形态可选：

1. 独立微服务。
2. 上层语音编排服务的本地模块。
3. Sidecar。

选择标准：

1. 能稳定加载多个 Persona 专属模型。
2. 支持短输出、低延迟和超时控制。
3. 支持灰度、回滚和模型版本隔离。
4. 高并发下 p99 延迟不明显劣化。
5. 能给出多个 Persona 专属模型同时加载时的内存占用估算，并明确采用全量常驻、按需加载还是多实例分片部署。

### 7.4 生成参数约束

上线配置必须限制：

1. 最大输出 token 数，避免生成完整回答。
2. temperature 和 top_p，优先保证稳定、自然和可控。
3. stop 规则，避免多句输出。
4. 推理超时预算，超时立即进入兜底路径。

### 7.5 延迟分解与压测

需要按 `route` 分别统计：

1. 请求解析耗时。
2. Query 归一化耗时。
3. Static Reply Gate 匹配耗时。
4. 缓存查询耗时。
5. Tokenizer 耗时。
6. MiniMind3 推理耗时。
7. 后处理校验耗时。
8. 安全校验耗时。
9. 兜底选择耗时。
10. 服务总响应耗时。

目标指标：

1. Static Reply Gate 包含归一化和规则匹配，建议 `p95 < 100μs`、`p99 < 500μs`。
2. MiniMind3 路径延迟需通过本地压测确定上线门槛，目标是显著低于云端 Qwen-32B 约 1 秒的调用延迟。
3. 整体服务延迟按 `static_reply`、`minimind3_filler`、`fallback_filler` 分别统计 p50/p95/p99。
4. 联调链路中，从 ASR 输出完整 Query 到垫话 TTS 首个可播放音频帧，建议 `p95 < 200ms`。
5. 高并发压测需覆盖目标 QPS、2 倍目标 QPS、突增流量、缓存命中率 0%/50%/90% 场景。

交付物：

1. 部署配置和模型加载方案。
2. 缓存实现。
3. 压测脚本。
4. 延迟分解报表。
5. 推荐部署规格，包括单模型和多 Persona 同时加载的内存预算。

---

## 8. 阶段七：下游联调与体验验收

### 8.1 下游集成约定

下游系统必须遵守：

1. 当 `filler_reply.continue_with_main_llm=false` 时，只播放 `choices[0].message.content`，不调用 Qwen-Plus。
2. 当 `filler_reply.continue_with_main_llm=true` 时，将 `choices[0].message.content` 作为已播放前缀传给 Qwen-Plus。
3. Qwen-Plus Prompt 必须要求主模型从此前缀之后自然续写，不得重复前缀。
4. 如果主模型输出仍包含同一前缀，下游需要在播放或合成前去重。
5. 本项目不返回音频帧，TTS 合成由下游负责。

### 8.2 联调测试用例

联调用例至少覆盖：

1. 问候、结束、感谢、确认、取消命中静态规则。
2. 静态规则反例不被误拦截，例如“谢谢你帮我写一封邮件”应进入 MiniMind3。
3. 每个 Persona 都能正确路由到专属模型。
4. Persona 模型不可用时进入兜底路径。
5. MiniMind3 输出过长、无标点、直接回答、事实污染时进入兜底路径。
6. Qwen-Plus 接收已播放前缀后不重复前缀。
7. TTS 可以立即播放 `choices[0].message.content`。

### 8.3 体验验收

人工评估维度：

1. 静态短回复是否自然、完整、符合 Persona。
2. 静态规则是否误拦截真实问题。
3. MiniMind3 前缀是否自然、短、适合 TTS。
4. 前缀是否没有直接回答问题。
5. 前缀与 Qwen-Plus 续写是否连贯。
6. 是否出现重复播放、语义割裂或语气不一致。

上线前建议达到：

1. 线上垫话自然度平均分 `>= 4.2 / 5`。
2. 线上续写连贯性通过率 `>= 95%`。
3. 重复与割裂率 `< 2%`。
4. `buffer_ready_rate >= 95%`（系统级集成指标，需与语音编排系统联合度量）。
5. 快路径首音频延迟 `p95 < 200ms`（系统级集成指标，需与 ASR、TTS 和播放链路联合验收）。

### 8.4 在线 A/B 实验

建议灰度指标：

1. 用户打断率。
2. 首轮继续对话率。
3. 负反馈率。
4. 平均感知等待时长。
5. `buffer_ready_rate`。
6. 各 `route` 分布。
7. 各 Persona 输出自然度和兜底率。
8. 静态规则命中率和误拦截抽检结果。

实验组应降低感知等待，同时不能显著提升打断率或负反馈率。

交付物：

1. 联调测试用例。
2. Qwen-Plus 续写 Prompt 约定。
3. 人工体验评估报告。
4. A/B 实验方案与指标看板。

---

## 9. 阶段八：灰度上线与迭代闭环

### 9.1 灰度发布顺序

推荐发布顺序：

1. 仅日志模式：服务返回结果但不播放，用于观察 `route` 分布、规则命中和模型输出。
2. 内部用户灰度：开启真实播放，收集主观反馈和联调问题。
3. 小流量线上灰度：按用户、场景或 Persona 分桶。
4. 分 Persona 放量：优先放量评估通过且兜底率低的 Persona。
5. 全量上线：保留规则、模型、兜底和总开关的快速回滚能力。

### 9.2 线上监控

必须监控：

1. 请求量、错误率、超时率。
2. `route` 分布：`static_reply`、`minimind3_filler`、`fallback_filler`。
3. p50、p95、p99 延迟，并按 `route`、Persona、模型版本分桶。
4. Static Reply Gate 命中率、规则级命中率和误拦截抽检结果。
5. MiniMind3 合法输出率、兜底率和 `fallback_reason` 分布。
6. 缓存命中率、容量使用率和淘汰次数。
7. Persona 模型可用性和模型版本分布。
8. 下游反馈的重复播放、语义割裂、用户打断和负反馈事件。

### 9.3 迭代闭环

建议每轮迭代流程：

1. 从线上日志采样静态规则误拦截、MiniMind3 不合法输出、兜底高频 Query 和用户负反馈样本。
2. 进入人工复核池。
3. 更新静态规则反例、兜底库和 hard cases 集。
4. 按 Persona 补充 SFT 数据。
5. 重新训练或微调对应 Persona 模型。
6. 离线评估通过后灰度发布新 `model_version`。
7. 对比新旧版本离线指标和在线指标。

交付物：

1. 灰度发布方案。
2. 线上监控看板。
3. 回滚预案。
4. 规则、模型、兜底库和 hard cases 的迭代流程。

---

## 10. 推荐里程碑

| 里程碑 | 主要内容 | 验收标准 |
| :--- | :--- | :--- |
| M1：契约固化 | OpenAI 兼容 API、枚举、下游续写约定、验收指标确认 | `messages`、`metadata`、`choices` 和 `filler_reply` 字段无歧义 |
| M2：静态规则 v1 | 规则库、归一化、反例、回归测试 | 覆盖寒暄/结束/感谢/确认/取消，误拦截率 `< 1%` |
| M3：SFT 数据集 v1 | 按 Persona 生成、校验、复核、切分数据 | 每个 Persona 数据可训练，hard cases 版本化 |
| M4：MiniMind3 模型 v1 | Persona 专属 SFT/LoRA、评估、导出 | 合法输出率、禁答率、事实污染率和人工评分达标 |
| M5：服务 v1 | Static Reply Gate、模型路由、校验、兜底、日志 | 三段式链路可联调，异常可降级 |
| M6：性能验收 | 本地部署、缓存、压测、延迟优化 | Static Reply Gate 与 MiniMind3 路径延迟分桶达标 |
| M7：体验验收 | 下游联调、TTS 播放、Qwen-Plus 续写、人工评估 | 自然度、连贯性、重复割裂率和首音频延迟达标 |
| M8：灰度上线 | 分 Persona 小流量发布与监控 | 指标稳定，可快速回滚 |

---

## 11. 上线检查清单

### 11.1 接口与契约

- [ ] `POST /v1/chat/completions` 兼容入口已定义。
- [ ] 请求字段 `model`、`messages`、`stream`、`metadata.persona_tag`、`metadata.locale`、`metadata.request_id` 已定义。
- [ ] `stream=true` 不支持时的错误码和 OpenAI 兼容错误响应已定义。
- [ ] Query 最大长度及超长处理策略已定义。
- [ ] 响应字段 `id`、`object`、`created`、`model`、`choices`、`usage`、`filler_reply` 已定义。
- [ ] `choices[0].message.content` 明确为当前要立即播放的文本。
- [ ] `filler_reply.route`、`filler_reply.reply_kind`、`filler_reply.fallback_reason` 枚举已固化。
- [ ] 下游确认 `filler_reply.continue_with_main_llm=false` 时不调用 Qwen-Plus。
- [ ] 下游确认 `filler_reply.continue_with_main_llm=true` 时将 `choices[0].message.content` 作为已播放前缀传给 Qwen-Plus。
- [ ] 下游具备重复前缀去重策略。

### 11.2 静态规则

- [ ] 静态规则只覆盖寒暄、结束、感谢、确认和取消类 Query。
- [ ] 每条规则配置 `rule_id`、匹配条件、候选短回复、适用 Persona、版本号、命中样例和反例。
- [ ] 每个规则候选短回复通过 TTS 口语自然性检查。
- [ ] 静态规则误拦截率 `< 1%`。
- [ ] 规则库支持版本化、灰度、热更新或快速回滚。
- [ ] 日志记录 `route=static_reply`、`rule_id` 和规则版本。

### 11.3 数据与模型

- [ ] 每个 Persona 的 SFT 数据集已版本化。
- [ ] 训练样本输入只包含 Query，输出只包含短前缀。
- [ ] `persona_tag` 不拼接进 MiniMind3 单次模型输入。
- [ ] 每个 Persona 都有独立模型版本。
- [ ] 模型版本命名使用三段式 `minimind3-filler-{persona-slug}-v{major}.{minor}.{patch}`。
- [ ] MiniMind3 合法输出率 `>= 98%`。
- [ ] 禁答违规率 `< 1%`。
- [ ] 事实污染率 `< 0.5%`。
- [ ] 前缀自然度平均分 `>= 4.2 / 5`。
- [ ] 续写连贯性通过率 `>= 95%`。
- [ ] hard cases 集回归通过。

### 11.4 服务与性能

- [ ] Query 归一化逻辑已实现并版本化。
- [ ] Query Prompt 注入防护已实现，包括长度截断、控制字符过滤和用户输入边界标记。
- [ ] Static Reply Gate 常驻内存并完成压测。
- [ ] Persona 模型路由失败不会静默使用其他 Persona 模型。
- [ ] 未知 `persona_tag` 会进入兜底路径并记录 `fallback_reason=model_unavailable`。
- [ ] MiniMind3 输出校验覆盖长度、标点、禁答、事实污染、操作承诺、安全和续写可行性。
- [ ] MiniMind3 超时时会丢弃部分输出，不播放不完整 token。
- [ ] 兜底短前缀库已版本化。
- [ ] MiniMind3 超时、不可用或输出不合法时返回 `route=fallback_filler`。
- [ ] 模型结果缓存 Key 包含归一化 Query、Persona、模型版本、静态规则版本和归一化逻辑版本。
- [ ] 静态规则或归一化逻辑热更新时具备缓存隔离或淘汰策略。
- [ ] 多 Persona 专属模型同时加载的内存预算已评估。
- [ ] Static Reply Gate 延迟建议 `p95 < 100μs`、`p99 < 500μs`。
- [ ] 整体服务延迟已按 `route` 分桶统计 p50/p95/p99。
- [ ] 高并发压测覆盖目标 QPS、2 倍目标 QPS 和突增流量。

### 11.5 联调与上线

- [ ] TTS 链路已完成首音频延迟验证。
- [ ] 快路径首音频延迟 `p95 < 200ms` 已与 ASR、TTS 和播放链路联合验收。
- [ ] `buffer_ready_rate >= 95%` 已与语音编排系统联合度量。
- [ ] 重复与割裂率 `< 2%`。
- [ ] A/B 实验指标看板已准备。
- [ ] 灰度开关和回滚方案已准备。
- [ ] 线上日志可按 `request_id` 回溯规则、模型、兜底和下游续写行为。
