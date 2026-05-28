# 极轻量高并发“垫话/过渡回复模型”需求与架构设计方案

在实时语音交互（如语音助手、智能客服、实时对话数字人）场景中，主大语言模型（Main LLM）的首字延迟（TTFT, Time to First Token）通常在数百毫秒至数秒之间，这会导致用户在说话结束后感到明显的停顿与迟钝。

为了消除这种“尴尬的沉默”，本项目拟构建一个**极轻量、高并发的“垫话/过渡回复模型”（Filler Reply Model）**。当用户结束说话后，系统先用静态规则处理高频寒暄、结束、感谢和确认类输入；其他 Query 则根据 Persona 调用本地部署的对应 MiniMind3 短前缀模型，快速生成一句符合当前意图与助理角色设定的自然垫话前缀，并即时通过 TTS 播放。主模型（如 Qwen-Plus）随后从该前缀之后继续回答，从而平滑对冲主模型首字延迟。

---

## 1. 数据定义与数据工程

本着“极轻量、低延迟”的原则，模型的数据流和输入输出进行如下精简设计。

### 1.1 输入数据定义
系统对外提供 OpenAI 兼容 API，推荐使用 `POST /v1/chat/completions`。决策阶段仍只依赖两个核心语义输入，**默认不依赖历史上下文和声学特征**：
1.  **当前用户 Query 文本**：通过 OpenAI Chat Completions 的 `messages` 传入，取最后一条 `role=user` 的 `content` 作为 ASR（语音识别）输出的当前轮完整文本（例如："你觉得人工智能未来会毁灭人类吗？"）。
2.  **助理角色分类标签（Persona Tag）**：通过请求级 `metadata.persona_tag` 传入，属于静态配置，用于选择静态回复候选和路由到对应 Persona 专属 MiniMind3 模型。例如：
    *   `male_white_collar`（中年男白领）
    *   `female_receptionist`（前台女助理）
    *   `grandpa`（老爷爷）
    *   `young_girl`（女童）

其中，静态规则主要由 Query 决定；Persona 不参与“是否命中静态回复规则”的判断，但会影响命中规则后的候选回复选择。MiniMind3 生成阶段不把 Persona 作为文本参数输入模型，而是由服务根据 `metadata.persona_tag` 路由到对应的 Persona 专属模型；单个 MiniMind3 模型只接收 Query 文本并输出该 Persona 风格下的短前缀。

输入约束：
*   **Query 长度上限**：v1 阶段需要配置最大输入长度（例如按字符数或 Token 数限制）。超过上限时，应按产品策略拒绝请求或截断到安全长度；截断必须记录日志，并避免破坏静态规则判断。
*   **Prompt 注入防护**：Query 作为用户输入嵌入 MiniMind3 推理 Prompt 前，需要做长度截断、控制字符过滤和边界标记，避免用户通过换行、伪造指令或特殊分隔符干扰“只生成短前缀”的模型行为。

### 1.2 输出数据定义
系统返回 OpenAI Chat Completions 兼容响应。`choices[0].message.content` 直接承载**当前要立即播放的文本**，并通过响应扩展字段显式标记后续是否需要主模型续写。

核心输出字段：
1.  **`choices[0].message.content`**：当前要立刻播放的文本。
    *   静态规则命中时，内容是完整短回复，例如“你好呀！”、“不客气！”、“好的。”。
    *   MiniMind3 路径命中时，内容是主回复的短前缀，例如“这个问题可以这样看，”、“我来帮你梳理一下，”。
2.  **`route`**：当前输出来源。
    *   `static_reply`：静态规则命中，返回固定或随机短回复。
    *   `minimind3_filler`：未命中静态规则，由 MiniMind3 生成短垫话前缀。
    *   `fallback_filler`：MiniMind3 超时或输出不合法，使用兜底短前缀。
3.  **`continue_with_main_llm`**：是否需要继续调用主模型。
    *   `false`：静态短回复已经足够完整，不需要 Qwen-Plus 续写。
    *   `true`：当前文本只是回复前缀，必须让 Qwen-Plus 从此前缀之后继续回答。
4.  **`reply_kind`**：文本语义形态，用于日志分析和下游编排。
    *   `complete_static_reply`：完整静态短回复。
    *   `filler_prefix`：可续写的垫话前缀。
5.  **`rule_id` / `model_version` / `fallback_reason`**：可选追踪字段，用于回溯线上行为。

业务元信息建议放在 OpenAI 响应外壳的顶层扩展字段 `filler_reply` 中，例如 `filler_reply.route`、`filler_reply.reply_kind`、`filler_reply.continue_with_main_llm`。OpenAI 客户端可以继续按标准路径读取 `choices[0].message.content`，需要编排信息的上游系统再读取 `filler_reply`。

输出设计原则：
*   **静态规则不是“不输出”**：寒暄、结束、感谢和确认类 Query 命中规则后，也应返回固定或随机短回复，只是不再调用 MiniMind3 和主模型。
*   **MiniMind3 只生成前缀，不回答问题**：生成内容必须短、自然、可被主模型继续续写，不得直接给出答案或引入事实。
*   **主模型必须感知已播放前缀**：`filler_reply.continue_with_main_llm=true` 时，下游需要把 `choices[0].message.content` 作为已播放前缀传给 Qwen-Plus，要求其从该前缀之后自然续写，且不得重复前缀。
*   **输出文本必须适合 TTS**：文本应口语自然、长度可控，边界清晰，优先以逗号、句号或感叹号结尾。

### 1.3 数据集建设与语料合成
为了训练 MiniMind3 短前缀模型，需要按 Persona 分别构建 SFT 数据集，并为每个 Persona 训练或微调一个独立的 MiniMind3 短前缀模型。每个 Persona 数据集内部样本结构为 `(Query, Filler_Prefix)`；Persona 只作为数据分组、模型版本和线上路由信息，不作为单次模型推理的输入文本。同时需要单独维护静态短回复规则库与兜底短前缀库。

*   **语料来源**：
    *   收集现有的高频用户问题列表，包括常见 QA、闲聊、任务型请求、开放观点、情绪表达、检索请求和指代不明请求。
    *   收集线上真实 Query，并按脱敏、去重、聚类后的结果进入训练候选集。
*   **静态规则数据**：
    *   覆盖寒暄、结束、感谢、确认、取消等无需主模型续写的 Query。
    *   每条规则配置 `rule_id`、匹配条件、候选短回复、适用 Persona、版本号和示例反例。
    *   候选短回复可以固定返回，也可以在同一 Persona 下随机选择。
*   **MiniMind3 SFT 数据**：
    *   使用 Qwen-32B、Qwen-Plus 或更强模型批量生成高质量短前缀样本。
    *   按 Persona 分别生成和维护训练样本；每个样本输入只包含用户 Query，输出只包含一句短垫话前缀。
    *   对同一 Query 生成多个 Persona 版本，但分别写入各自 Persona 的训练集，用于训练多个风格固定的专属模型。
    *   对高频 Query 做同义改写，扩展泛化能力。
*   **输出约束校验**：
    *   输出长度建议控制在 5-18 个中文字符，特殊情况下不超过 25 个中文字符。
    *   输出必须以 `，`、`。` 或 `！` 结尾。
    *   输出不得直接回答问题，不得编造事实，不得承诺未发生的检索或操作。
    *   输出必须能作为主回复第一句话前缀，被 Qwen-Plus 自然续写。
*   **人工复核与 hard cases**：
    *   对模型生成过长、重复、答非所问、风格不符、无法续写的样本进入人工复核。
    *   建立 hard cases 集，每次 SFT 或 LoRA 迭代后都必须回归测试。

---

## 2. 模型选型评估

针对高并发、低延迟、中文语义泛化和自然表达的要求，我们对以下几类方案进行评估：

| 评估维度 | 方案 A: 分类器 + 模板随机抽取 | 方案 B: 直接调用云端大模型生成 | 方案 C: MiniMind3 SFT/LoRA 短前缀生成 |
| :--- | :--- | :--- | :--- |
| **推理延迟** | 极低，通常只需规则匹配和模板检索 | 较高，云端 Qwen-32B 单次调用约 1 秒 | 可控，需本地量化和短输出约束 |
| **表达自然度** | 中等，容易重复和机械 | 很好 | 较好，可通过 SFT 学习自然承接 |
| **Query 泛化能力** | 有限，模板无法覆盖千变万化的 Query | 很强 | 较强，取决于 SFT 数据质量 |
| **角色风格控制** | 依赖模板库维护 | 较强 | 较强，可通过 Persona 专属数据和专属模型控制 |
| **线上成本** | 极低 | 较高，且依赖外部服务延迟 | 低，适合本地部署和高并发 |
| **主要风险** | 生硬、重复、无法贴合语境 | 延迟高、成本高、链路不稳定 | 可能生成过长、直接回答或不适合续写 |

### 最终选型决策：**静态短回复规则 + MiniMind3 SFT/LoRA 短前缀生成 + Qwen-Plus 承接续写**

*   **静态规则层**：只处理寒暄、结束、感谢、确认、取消等高频且无需复杂语义理解的输入，返回固定或随机完整短回复，不调用 MiniMind3 和 Qwen-Plus。
*   **MiniMind3 层**：负责除静态规则外的大多数 Query，生成一句短、自然、可续写的垫话前缀。该层按 Persona 部署多个专属 MiniMind3 模型，每个模型通过对应 Persona 的 SFT/LoRA 数据从 Qwen-32B、Qwen-Plus 或更强模型的高质量样本中蒸馏能力。
*   **Qwen-Plus 层**：负责真正回答用户问题，并且必须接收“已播放前缀”，从此前缀后继续生成，避免重复和割裂。
*   **兜底层**：MiniMind3 超时、输出不合法或安全校验失败时，使用固定兜底前缀，例如“我想一下，”、“我来帮你梳理一下，”。

---

## 3. 特征工程设计

为了保证端到端极速响应，特征工程需进行极致剪裁。

### 3.1 文本特征设计
不进行繁琐的分词、词性标注等前置 CPU 密集型操作，而是直接利用 MiniMind3 的 Tokenizer 进行处理：
*   **静态规则输入**：仅使用归一化后的 Query 文本，不使用 Persona 做命中判断。
    *   示例：`谢谢`、`好的`、`再见`。
    *   命中后可根据 Persona 从候选短回复中选择更贴合角色的版本。
*   **MiniMind3 模型路由**：根据 Persona Tag 选择对应的 Persona 专属 MiniMind3 模型。
    *   示例：`female_receptionist` 路由到 `minimind3-filler-female-receptionist-v1.0.0`。
    *   路由失败或目标模型不可用时，直接进入兜底短前缀路径，避免错误使用其他 Persona 模型导致风格错配。
    *   未知 `persona_tag`（请求值存在但不在路由表中）应视为路由失败，进入 `fallback_filler` 并记录 `fallback_reason=model_unavailable` 和告警；不得静默映射到其他 Persona 模型。
*   **MiniMind3 输入**：只使用 Query 文本，不拼接 Persona Tag。
    *   建议拼接格式：`用户：{query}\n请生成一句可续写的短垫话前缀：`
    *   示例输入：`用户：我最近真的很焦虑，不知道怎么办\n请生成一句可续写的短垫话前缀：`
    *   示例输出：`我能理解你现在的压力，`
*   **主模型输入**：Qwen-Plus 需要接收原始 Query、Persona、以及 MiniMind3 已播放的前缀。
    *   要求主模型从该前缀之后继续回答，不要重复此前缀。

### 3.2 排除的特征说明
*   **无声学特征**：不引入 VAD 停顿时间、说话人语速、音量波动等声学特征，避免前置音频解析带来的延迟，且降低系统复杂度。
*   **无复杂上下文特征**：v1 阶段默认只使用当前轮 Query，不拼接历史对话，避免输入膨胀拖慢推理。后续如需处理“刚才那个”“这个怎么办”等强指代问题，可单独设计短上下文摘要特征。

---

## 4. 推理流程设计

线上推理采用**静态短回复规则 + MiniMind3 短前缀生成 + Qwen-Plus 承接续写**的三段式机制。该设计的目标是用静态规则保证高频闲聊场景的极低延迟，用 MiniMind3 提升垫话自然度，用 Qwen-Plus 完成真正的内容回答。

### 4.1 前置规则引擎：Static Reply Gate

用户 ASR 输出完整 Query 文本后，系统首先进行轻量归一化（如去除首尾空白、统一全半角、规整常见标点），然后进入静态回复规则表匹配。该阶段只做内存哈希、前缀/正则或少量规则判断，不调用 Tokenizer 和模型，整体耗时目标为**亚毫秒级**（p95 < 100μs，详见 6.2 节）。

Static Reply Gate 主要覆盖以下场景：
*   **问候**：如“你好”“您好”“哈喽”“嗨”。
*   **结束**：如“再见”“拜拜”“下次聊”。
*   **感谢**：如“谢谢”“多谢”“辛苦了”。
*   **确认**：如“好的”“好”“嗯”“行”“可以”。
*   **取消/否定**：如“不用了”“算了”“没事了”。

静态规则命中时，规则引擎直接返回固定或随机完整短回复，不再进入 MiniMind3 和 Qwen-Plus。例如“谢谢”可以返回“不客气！”；“再见”可以返回“下次聊！”；“好的”可以返回“好的。”。

### 4.2 三段式路由策略

推理链路分为三类路径：
1.  **静态短回复路径（Static Reply Path）**：Static Reply Gate 命中后立即返回完整短回复，`continue_with_main_llm=false`。
2.  **MiniMind3 短前缀路径（Filler Prefix Path）**：规则未命中时，请求流转至 MiniMind3，由模型生成可续写短前缀，`continue_with_main_llm=true`。
3.  **兜底短前缀路径（Fallback Path）**：MiniMind3 超时、输出不合法或安全校验失败时，返回预设兜底短前缀，`continue_with_main_llm=true`。

整体流程如下：
1.  ASR 输出当前轮完整 Query。
2.  对 Query 做轻量文本归一化。
3.  查询 Static Reply Gate。
4.  若命中规则，返回固定或随机完整短回复，并结束本轮链路。
5.  若未命中规则，根据 Persona Tag 选择对应的 MiniMind3 专属模型，并调用该模型生成短垫话前缀。
6.  对 MiniMind3 输出做长度、标点、禁答、安全和续写可行性校验。
7.  若校验通过，立即播放该短前缀；若校验失败，使用兜底短前缀。
8.  将已播放前缀传给 Qwen-Plus，要求主模型从该前缀之后继续回答。

### 4.3 规则与模型的边界

Static Reply Gate 只处理**高置信、高频、低歧义、无需主模型续写**的 Query，不承担复杂语义泛化。凡是任务请求、开放问题、事实问题、推理问题、情绪表达、检索请求、指代不明请求，都默认下沉到 MiniMind3 生成短前缀。

为避免规则膨胀导致维护成本和误拦截风险，静态规则需要满足：
*   每条规则都有明确的业务动机、命中样例和反例。
*   规则变更需要版本化，并记录线上命中率、误拦截样本、候选回复和回滚信息。
*   规则优先级高于 MiniMind3，但不得覆盖边界模糊样本。
*   边界样本应进入 hard cases 数据集，用于后续 MiniMind3 SFT/LoRA 迭代。

### 4.4 MiniMind3 输出校验

MiniMind3 的输出必须通过后处理校验后才能播放：
*   **长度校验**：中文长度建议 5-18 字，最长不超过 25 字。
*   **边界校验**：必须以 `，`、`。` 或 `！` 结尾。
*   **禁答校验**：不得直接回答用户问题，例如用户问“苹果英文怎么说”，MiniMind3 不应输出“苹果英文是 apple。”。
*   **事实校验**：不得编造具体时间、地点、人物、数据或外部事实。
*   **操作承诺校验**：不得承诺未执行的操作，例如“我查到了，”或“我已经帮你订好了，”。
*   **续写校验**：文本必须能被 Qwen-Plus 从后面自然接上。
*   **安全校验**：不得输出冒犯、歧视、露骨或其他不适合 TTS 播放的内容。v1 阶段应优先采用本地轻量规则或分类器，避免外部审核链路破坏低延迟目标；若接入外部安全服务，必须单独纳入延迟预算。
*   **超时处理**：MiniMind3 推理超时时，无论是否已经产生部分 token，都必须丢弃部分输出并进入兜底路径，不对不完整输出做播放或校验。

---

## 5. 项目边界与服务接口

本项目只负责“给定用户 Query 和 Persona，选择合适的静态回复或 Persona 专属 MiniMind3 模型，返回当前应立即播放的文本，并声明是否需要主模型续写”。ASR、TTS、主 LLM 调用、音频拼接、播放策略和多路并发编排属于上层语音交互系统，不作为本项目的设计范围。

### 5.1 核心输入
服务暴露 OpenAI 兼容接口：

```text
POST /v1/chat/completions
```

请求体遵循 Chat Completions 格式，并通过 `metadata` 携带本项目所需的非文本路由信息：

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
    "request_id": "req_123"
  }
}
```

字段要求：
1.  **`model`**：必填，OpenAI 兼容模型名。建议使用服务级模型名，例如 `filler-reply-minimind3`；真实 Persona 专属 MiniMind3 模型由服务内部根据 `metadata.persona_tag` 路由。
2.  **`messages`**：必填，兼容 OpenAI Chat Completions。服务取最后一条 `role=user` 的 `content` 作为当前轮 Query；v1 阶段默认不拼接历史上下文。
3.  **`stream`**：可选，初版建议仅支持 `false`。垫话文本很短，不需要流式输出；当客户端传入 `stream=true` 且服务尚未实现流式兼容时，应返回 OpenAI 兼容错误响应（建议 HTTP 400 或 501），不得静默忽略。
4.  **`metadata.persona_tag`**：必填，当前助理角色标签，用于静态短回复选择和 MiniMind3 专属模型路由，不作为 MiniMind3 的文本输入参数。
5.  **`metadata.locale`**：可选，默认为 `zh-CN`，用于后续支持多语言静态回复和短前缀生成。
6.  **`metadata.request_id`**：可选，用于链路追踪和日志关联。
7.  **Query 长度**：服务需要定义并校验最大 Query 长度，超过上限时应返回错误或按配置截断，并在响应或日志中记录处理方式。

### 5.2 核心输出
服务响应保持 OpenAI Chat Completions 外壳。当前要播放的文本放在 `choices[0].message.content`；业务编排元信息放在顶层扩展字段 `filler_reply`。

示例响应：

```json
{
  "id": "chatcmpl-req_123",
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
    "request_id": "req_123"
  }
}
```

静态短回复示例：

```json
{
  "id": "chatcmpl-req_124",
  "object": "chat.completion",
  "created": 1779979681,
  "model": "filler-reply-minimind3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "不客气！"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "filler_reply": {
    "reply_kind": "complete_static_reply",
    "continue_with_main_llm": false,
    "route": "static_reply",
    "rule_id": "thanks.reply.001",
    "model_version": null,
    "fallback_reason": null,
    "request_id": "req_124"
  }
}
```

`usage` 字段说明：为了保持 OpenAI 兼容外壳，响应应稳定返回 `usage`。MiniMind3 路径可以返回实际或估算 token；静态规则路径没有模型推理，建议统一返回 `0`，或按服务级统计口径估算，但同一部署内必须保持一致。

`filler_reply` 字段说明：
1.  **`reply_kind`**：文本形态，取值为 `complete_static_reply` 或 `filler_prefix`。
2.  **`continue_with_main_llm`**：是否需要继续调用 Qwen-Plus 续写。
3.  **`route`**：推理路径，取值为 `static_reply`、`minimind3_filler` 或 `fallback_filler`。
4.  **`rule_id`**：静态规则命中时返回规则 ID；未命中时为空。
5.  **`model_version`**：MiniMind3 模型版本，用于灰度、回滚和离线分析；若为静态规则命中，可为空。
6.  **`fallback_reason`**：兜底路径触发原因，例如 `timeout`、`model_unavailable`、`invalid_length`、`invalid_boundary`、`direct_answer`、`unsafe_output`。
7.  **`request_id`**：用于链路追踪和日志关联，优先透传 `metadata.request_id`。

### 5.3 下游集成约定
虽然本项目不负责主 LLM 和 TTS 编排，但输出需要满足下游集成约束：
1.  **静态短回复是完整回复**：当 `filler_reply.continue_with_main_llm=false` 时，下游只需播放 `choices[0].message.content`，不再调用 Qwen-Plus。
2.  **垫话文本是主回复前缀**：当 `filler_reply.continue_with_main_llm=true` 时，`choices[0].message.content` 应被下游作为主模型回复的已播放前缀，主模型需要从该前缀后继续生成。
3.  **前缀边界固定**：`filler_reply.reply_kind=filler_prefix` 时，`choices[0].message.content` 必须以逗号、句号或感叹号结尾，保证可以独立播放。
4.  **避免重复播放**：如果下游主模型输出中包含同一前缀，下游需要在播放或合成前去重。
5.  **不返回音频**：本项目只返回文本和元信息，不负责 TTS 合成，也不返回音频帧。

### 5.4 部署与缓存
1.  **引擎选择**：MiniMind3 优先使用本地量化推理方案部署，按实际工程栈选择 llama.cpp、vLLM、ONNX Runtime、TensorRT-LLM 或其他轻量推理引擎。
2.  **服务形态**：可独立部署为微服务，也可作为上层语音编排服务的本地模块或 Sidecar。
3.  **Static Reply Gate 部署**：
    *   静态规则常驻内存，随服务启动加载，内容包括规则 ID、匹配条件、候选短回复、Persona 适配信息和版本号。
    *   映射 Key 应优先使用归一化后的 Query；Persona 不参与命中判断，但会影响命中规则后的候选短回复选择。
    *   规则表需要支持热更新或灰度发布，并在日志中记录 `route=static_reply`、`rule_id` 和规则版本。
4.  **模型结果缓存**：
    *   对未命中静态规则但线上高频重复的 Query，可建立内存级 KV 缓存，缓存内容为 MiniMind3 生成的短前缀和校验结果，命中缓存时跳过模型推理。
    *   缓存 Key 应包含归一化后的 Query、Persona Tag、实际命中的 MiniMind3 模型版本、静态规则版本和归一化逻辑版本。
    *   静态规则或归一化逻辑热更新时，需要按版本隔离缓存或淘汰受影响缓存，避免新规则下本应命中的 Query 继续复用旧 MiniMind3 结果。
    *   缓存值需要记录生成时间和命中次数，便于发现高频不自然前缀并进入人工复核。
    *   部署规格需要估算多个 Persona 专属模型同时加载的内存占用，并明确是否采用全量常驻、按需加载或多实例分片。
5.  **生成参数约束**：
    *   最大输出 token 数必须严格限制，避免小模型生成完整回答。
    *   建议使用低温度或有限随机性，优先保证稳定、自然和可控。
    *   超时预算应小于主模型 TTFT 的可感知窗口，超时立即进入兜底路径。

---

## 6. 效果评估与验收标准

为了保证该模型可以稳定服务实时语音交互，评估需要同时覆盖静态规则准确性、MiniMind3 生成质量、在线延迟、语音衔接体验和兜底质量。

### 6.1 离线生成指标
1.  **静态规则准确性**：
    *   静态规则误拦截率建议 `< 1%`。一旦把真实问题误判成寒暄/确认，会导致主模型不回答。
    *   每条规则必须有命中样例和反例，并对线上高频命中进行抽检。
2.  **MiniMind3 合法输出率**：
    *   长度、标点、安全、禁答和续写校验综合通过率建议 `>= 98%`。
    *   不合法输出必须记录原始 Query、Persona、模型输出和失败原因。
3.  **前缀自然度**：
    *   人工评分 1-5 分，建议平均分 `>= 4.2`。
    *   重点评估是否口语自然、是否符合 Persona、是否突兀、是否过度重复。
4.  **禁答率与事实污染率**：
    *   MiniMind3 不应直接回答用户问题，禁答违规率建议 `< 1%`。
    *   不应生成具体事实、时间、数据或未执行操作，事实污染率建议 `< 0.5%`。
5.  **续写连贯性**：
    *   将 MiniMind3 前缀输入 Qwen-Plus 后，人工评估整体回答是否自然连贯，通过率建议 `>= 95%`。
    *   对不连贯样本建立 hard cases 集，后续每次模型迭代都必须回归测试。

### 6.2 在线延迟指标
1.  **Static Reply Gate 延迟**：包含 Query 归一化和规则匹配，建议 `p95 < 100μs`，`p99 < 500μs`。
2.  **MiniMind3 路径延迟**：包含 Tokenizer、模型推理、后处理，需通过本地压测确定上线门槛；目标是显著低于云端 Qwen-32B 约 1 秒的调用延迟。
3.  **整体服务延迟**：按 `route` 分别统计 `static_reply`、`minimind3_filler` 与 `fallback_filler` 的 p50/p95/p99，避免整体平均值掩盖模型路径问题。
4.  **快路径首音频延迟**（系统级集成指标）：从 ASR 输出完整 Query 到垫话 TTS 首个可播放音频帧，建议 `p95 < 200ms`。本项目只输出文本，该指标需与上游语音编排系统联合度量。
5.  **拼接等待指标**（系统级集成指标）：垫话播放结束时，主模型续写音频缓冲区应已经有可播放音频帧；建议统计 `buffer_ready_rate`，上线门槛 `>= 95%`。该指标涉及 TTS 和音频拼接，需与上游系统配合验收。
6.  **吞吐能力**：按目标并发量压测 `QPS`、CPU 使用率和内存占用，确保 p99 延迟不因高并发明显劣化。

### 6.3 体验质量指标
以下指标关注线上真实用户的端到端体验，与 6.1 节离线批量评估互为补充。
1.  **线上垫话自然度**：定期对线上真实 Query 的垫话输出进行人工抽检评分（1-5 分），建议平均分 `>= 4.2`（与 6.1 离线基线对齐），重点关注真实分布下的 Persona 匹配度和口语自然度。
2.  **线上续写连贯性**：定期抽检线上垫话前缀与主模型实际续写拼接后的完整回答，人工评估语义连贯通过率建议 `>= 95%`（与 6.1 离线基线对齐）。
3.  **重复与割裂率**：统计用户听到重复垫话、语义断裂、语气不一致的比例，建议 `< 2%`。
4.  **线上 A/B 指标**：观察用户打断率、首轮继续对话率、负反馈率、平均感知等待时长。实验组相比基线应降低感知等待，同时不能显著提升打断率或负反馈率。

### 6.4 静态回复与兜底库验收标准
1.  静态规则只覆盖寒暄、结束、感谢、确认和取消类 Query，不扩展到复杂问题，避免规则膨胀。
2.  每个静态规则至少配置 3-10 条候选短回复，并按 Persona 做必要区分。
3.  兜底短前缀必须短、通用、可续写，例如“我想一下，”“我来帮你梳理一下，”。
4.  静态回复和兜底前缀都需要版本化，线上日志应记录 `rule_id`、`route`、`model_version` 和 `fallback_reason`，便于回溯问题样本和做效果分析。
