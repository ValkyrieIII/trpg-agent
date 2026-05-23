# TRPG Agent v2 设计说明 — 交互模型反转

## 背景与目标

当前项目 v1 的交互逻辑是反的：AI 扮演角色「艾琳」与玩家对话，GM 只做背景叙事。正统 TRPG 中，玩家操控艾琳，AI 负责 GM（叙事/检定）+ NPC 扮演。

核心改动：交互模型反转，其余组件（骰子、检定、记忆、RAG、状态机）保持不动。

## 现状与约束

- 已有：骰子系统、检定引擎、状态机、ChromaDB 记忆、RAG 知识库、Rich CLI、LLM 封装（OpenAI SDK → DeepSeek）
- NPC 角色卡与玩家角色卡同结构（属性/技能 + 扮演层 personality/few_shot）
- NPC 由 GM 动态生成，自动持久化到 ChromaDB
- 玩家首次互动建卡，之后加载已有角色卡
- 一天开发时间

## 方案对比

### 方案一：GM 全权调度（选用）

GM agent 接收玩家输入，决定「叙事/检定/NPC 回应」，按需调用 NPC agent。

- 优点：GM 掌控节奏，NPC 人格通过独立 system prompt 保证稳定
- 缺点：每轮 1-2 次 LLM 调用

### 方案二：意图路由 + 独立管线

轻量分类器先判断输入类型，直接路由到对应管线，无 GM 中枢。

- 优点：纯对话省 token
- 缺点：叙事割裂，意图可能误判

### 方案三：全量上下文多 Agent

GM 和所有 NPC 合在一次 LLM 调用中。

- 优点：单次调用快
- 缺点：NPC 人格被稀释（就是 v1 的问题）

## 推荐方案

方案一。GM 作为叙事中枢保持连贯性，NPC 独立 agent 保证角色一致性。

---

## 详细设计

### 架构

```
玩家（扮演艾琳）
    │
   输入（行动/对话）
    │
    ▼
┌──────────────────────┐
│     GM Agent         │
│  叙事中枢 + 检定裁判   │
│                      │
│  1. 识别意图          │
│  2. 判断检定          │
│  3. 决定谁回应        │
└──────┬───────────────┘
       │
┌──────┼──────────┐
▼      ▼          ▼
GM叙事  检定引擎   NPC回应
(自己)  (已有)    (独立调用)
                  │
         ┌────────┼────────┐
         ▼        ▼        ▼
      NPC1     NPC2    ...动态NPC
     (独立角色卡，含人设/语气/few-shot)
```

**保持不变：** 骰子、检定引擎、状态机、ChromaDB 记忆、RAG 知识库、Rich CLI、LLM 封装

**改造/新增：** GM agent prompt 重写、NPC 管理模块、config 结构调整、main.py 交互文案

### 配置结构

`config.yaml` 改为定义玩家角色 + 世界设定。NPC 角色卡移到 `data/npcs/` 目录。

```yaml
# config.yaml
player:
  name: "艾琳"
  core: ["北方荒原的游侠", "曾独自在荒野生存十年"]
  attributes:
    strength: 14 / agility: 18 / intelligence: 12 / willpower: 15
  skills:
    - { name: "追踪", value: 75 }
    - { name: "弓箭", value: 80 }
    - { name: "野外生存", value: 90 }

world:
  name: "北境荒原"
  description: "被永冻魔法笼罩的辽阔土地..."
```

```yaml
# data/npcs/酒馆老板.yaml
name: "老马"
core: ["北境小镇酒馆老板", "年轻时是冒险者"]
attributes: { strength: 13, agility: 10, intelligence: 14, willpower: 12 }
skills:
  - { name: "交涉", value: 70 }
personality:
  tone: "热情粗犷，爱讲故事"
  verbal_tics: "每句话结尾加'是吧'"
  emotion_map:
    anger: "拍桌子，嗓门变大"
    trust: "压低声音分享秘密"
few_shot:
  - { input: "最近有什么新鲜事？", output: "嘿，来得正好！矿洞又传出怪声了，是吧。" }
```

### 关键组件

**`character.py`：** 基本不变，去掉 personality/few_shot 字段（这些归 NPC），保留属性/技能/加载。

**`npc.py`（新增）：** NPC 管理模块。
- `NPCCharacter` — 与 `Character` 同结构 + 扮演层（personality/few_shot），加载 YAML
- `NPCStore` — ChromaDB 持久化，支持语义检索、按名查找、save/load
- GM 可调用 `NPCStore.create(name, core, ...)` 动态创建 NPC

**`game_master.py`（改造）：** GM agent prompt 重写，调度逻辑调整。

新的 GM system prompt：
```
你是 TRPG 地下城主。玩家扮演{player_name}。
职责：
1. 叙述场景（第三人称客观描述）
2. 判断行动是否需要检定，执行检定
3. 决定哪个 NPC 回应玩家（可能无人回应，由你直接叙述）

规则：
- 玩家用括号声明行动，如"(我拔出短刀)"。你需要判断是否需要检定。
- 纯角色扮演动作（微笑、点头）不需要检定。
- 玩家直接对 NPC 说话时，判断该 NPC 是否在场。
- 不要替玩家角色说话或替ta做决定。
- 每段叙述不超过 150 字。
```

GM 调度流程：
```
玩家输入
  → GM Agent 分析（传入场景 NPC 列表 + 检定规则）
  → GM 返回结构化响应：{narration, check_result, responding_npc}
  → 如果 responding_npc 不为空：
      → 调用 NPC Agent（该 NPC 角色卡作为 system prompt）
      → 拼接：[GM] narration + [NPC名] npc_reply
  → 如果 responding_npc 为空：
      → 返回 [GM] narration
  → 记忆写入 + 状态更新（同 v1）
```

### 数据流：一轮对话

```
玩家输入: "老板，有没有见过可疑的人？"
    │
    ▼
GM Agent（传入：当前场景已知 NPC 列表 [老马, ...]）
    │
    GM 判断：玩家对酒馆老板说话 → responding_npc = "老马"
    检定判断：无
    叙事：酒馆里烟雾缭绕，老马擦着杯子抬起头。
    │
    ▼
NPC Agent（system prompt = 老马角色卡）
    输入 = "玩家对你说：老板，有没有见过可疑的人？"
    │
    输出 = "哈！你算问对人了。昨晚上有个戴兜帽的家伙..."
    │
    ▼
输出：
[GM] 酒馆里烟雾缭绕，老马擦着杯子抬起头。
[老马] 哈！你算问对人了。昨晚上有个戴兜帽的家伙...
    │
    ▼
记忆写入 + 状态更新
```

### 异常与边界处理

- **在场 NPC 列表维护：** GM 每次叙事时自动追踪当前场景有哪些 NPC，存储在 `self.scene_npcs: list[str]`
- **NPC 不在场：** GM 检测到玩家对不在场 NPC 说话时，叙述"ta 不在这里"或建议行动
- **无匹配 NPC：** 玩家对话没有明确指向任何 NPC 时，GM 直接叙事回应
- **NPC 动态创建：** GM 叙事中引入新 NPC 时，自动调用 NPCStore 创建并持久化
- **同 NPC 多轮对话：** NPC agent 调用时传入该 NPC 的独立对话历史（从 NPCStore 维护的 per-NPC history 中获取）
- **已有异常处理保留：** LLM 重试、检索无结果、角色卡校验等沿用 v1

### 测试策略

| 层 | 测什么 | 工具 |
|---|--------|------|
| NPC 加载/保存 | YAML 解析、ChromaDB 存取 | pytest |
| GM prompt | 意图识别准确性、结构化输出解析 | 手动 smoke |
| GM → NPC 调度 | 多 NPC 场景下选择正确 NPC 回应 | 手动 smoke |
| NPC 人格一致性 | 同 NPC 多轮对话语气稳定 | 手动 smoke |
| 已有系统 | 骰子/检定/状态机/记忆 — 不受影响 | 已有 pytest |

## 风险与待确认项

- GM 结构化输出（JSON）的可靠性 — DeepSeek V4 Flash 的 JSON mode 表现待验证，必要时降级为正则解析
- NPC per-agent 对话历史的上下文长度管理 — 需要类似 v1 的滑动窗口
- 场景 NPC 列表由 GM 自主维护，可能遗漏或遗忘 — 后续可考虑结构化追踪
