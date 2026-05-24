# 多 Agent 编排架构

> 基于 OpenAI Agents SDK (`openai-agents`) 的 TRPG 多 Agent 系统设计文档。
> 最后更新：2026-05-24

---

## 目录

1. [架构概述](#架构概述)
2. [核心设计决策](#核心设计决策)
3. [Agent 体系](#agent-体系)
4. [工具系统](#工具系统)
5. [记忆系统](#记忆系统)
6. [上下文管理](#上下文管理)
7. [数据流](#数据流)
8. [文件索引](#文件索引)

---

## 架构概述

```
                         ┌─────────────────────────────┐
                         │       Player Input          │
                         └─────────────┬───────────────┘
                                       │
                         ┌─────────────▼───────────────┐
                         │     GameMaster.process()     │
                         │                              │
                         │  ┌──────────────────────┐   │
                         │  │ _build_input()        │   │
                         │  │  记忆检索 + 知识查询   │   │
                         │  │  + 历史拼装            │   │
                         │  └──────────┬───────────┘   │
                         │             │               │
                         │  ┌──────────▼───────────┐   │
                         │  │ Runner.run_sync()     │   │
                         │  │ (SDK 自动管理工具循环) │   │
                         │  │                       │   │
                         │  │  GM Agent             │   │
                         │  │   ├─ roll_dice        │   │
                         │  │   ├─ difficulty_check │   │
                         │  │   ├─ combat_attack    │   │
                         │  │   ├─ invoke_npc ──────┼──► NPC Agent Pool
                         │  │   ├─ create_npc       │   │
                         │  │   ├─ search_memory    │   │
                         │  │   └─ ...              │   │
                         │  └──────────┬───────────┘   │
                         │             │               │
                         │  ┌──────────▼───────────┐   │
                         │  │ 记忆写入 + 历史裁剪    │   │
                         │  │ 世界模拟触发           │   │
                         │  └──────────────────────┘   │
                         └─────────────┬───────────────┘
                                       │
                         ┌─────────────▼───────────────┐
                         │    Response (叙述+建议)      │
                         └─────────────────────────────┘
```

**改造前 vs 改造后**：

| 维度 | 改造前 | 改造后 |
|------|--------|--------|
| 工具调用 | GM 输出 JSON `tool_calls`，手动分发 | SDK native function calling，自动循环 |
| 循环管理 | 手动 while 循环 + messages 拼接 (~130行) | `Runner.run_sync()` 自动管理 |
| NPC 回应 | `_tool_npc_speak` → `LLM.chat()` 被动生成 | NPC Agent 独立推理，自主决定如何回应 |
| 世界模拟 | 硬编码每 3 回合触发 | 保留外部触发，但内部可用 NPC Agent |
| 状态注入 | `self.xxx` 直接访问 GameMaster 属性 | `RunContextWrapper[GameContext]` 统一注入 |

---

## 核心设计决策

### 1. Orchestrator + Agents-as-Tools（非 Handoffs）

**选择原因**：GM 需要综合所有 NPC 反应后统一叙述，而非把控制权交给某个 NPC。

```
GM 调用 invoke_npc(name="老马", prompt="...")
  → 内部路由到 NPC Agent
  → NPC Agent 独立 LLM 推理
  → 返回回应给 GM
  → GM 综合叙述
```

对比 Handoffs 模式：

| 维度 | Handoffs（交接） | Agents-as-Tools（工具化） |
|------|-----------------|--------------------------|
| 控制流 | GM 交出控制权 | GM 始终保持控制 |
| 多 NPC 互动 | 困难（一次只能交接给一个 Agent） | 自然（GM 可调用多个 NPC tool） |
| 适用场景 | 客服路由、专业分诊 | **TRPG 叙事编排** |

### 2. 统一 NPC 调用入口

**不为每个 NPC 创建独立 tool**，而是提供统一的 `invoke_npc(name, prompt)`：

- GM 的 tools 列表固定（~13 个），NPC 动态增减无需修改 GM Agent
- GM 不需要记住每个 NPC 的工具名
- 避免了 SDK 动态修改 tools 列表的兼容性风险

### 3. 状态注入：GameContext

所有工具通过 `RunContextWrapper.context` 访问游戏状态，不依赖 `self.xxx`：

```python
@dataclass
class GameContext:
    player_state: StateMachine      # 玩家 HP/情绪/信任度/体力
    npc_store: NPCStore             # NPC 持久化与缓存
    memory: MemoryStore             # 主记忆 + NPC记忆
    knowledge: KnowledgeBase        # 世界知识库
    scene_npcs: list[str]           # 当前场景 NPC
    time_of_day: str                # 时间
    weather: str                    # 天气
    player_name: str                # 玩家名称
    player_card: str                # 角色卡摘要
    player_skills: list[dict]       # [{name, value}, ...]
    player_attributes: dict         # {力量, 敏捷, ...}
    npc_agents: dict[str, Agent]    # NPC Agent 池
    history_messages: list[dict]    # 对话历史
    llm: Any                        # 旧 LLM 实例（Phase 1 回退用）
    game_over: bool                 # 游戏结束标志
```

### 4. NPC Agent 的动静分离

- **instructions（静态）**：性格、说话风格、口头禅、few_shot — 创建后不变
- **input（动态）**：当前情绪、体力、场景、相关记忆 — 每次调用时注入

```python
# 静态：Agent 创建时
npc_agent = Agent(
    name="NPC_老马",
    instructions="你是老马，酒馆老板。说话粗豪但热心...",
    tools=[search_npc_memory],
)

# 动态：每次调用时
npc_input = build_npc_input(prompt, npc_context, npc_history)
result = Runner.run_sync(npc_agent, input=npc_input, context=npc_context)
```

### 5. 对话历史手动管理

`Runner.run_sync()` 是无状态的，不会自动记住之前的对话。在 `GameMaster` 中手动维护：

```python
self.history_messages: list[dict]  # [{"role": "user", "content": "..."}, ...]
self._trim_history()               # 保留最近 5 轮 (10 条消息)
```

### 6. 世界模拟保留外部触发

不交给 GM 自主判断（保持"意外感"），保留每 N 轮的定时触发。

### 7. 增量改造策略

- **Phase 1**：SDK Runner + 游戏工具迁移，`invoke_npc` 用旧 `LLM.chat` 逻辑
- **Phase 2**：`invoke_npc` 升级为真正的 NPC Agent 路由

### 8. DeepSeek 兼容性

DeepSeek 不支持 OpenAI Responses API，需强制使用 Chat Completions API：

```python
client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
set_default_openai_client(client)
set_default_openai_api("chat_completions")  # 关键配置
```

---

## Agent 体系

### GM Agent

**定义**：[agent_config.py](../trpg_agent/agent_config.py)

```python
Agent(
    name="GameMaster",
    instructions=build_gm_instructions(world_state),  # 动态更新
    tools=[
        roll_dice, difficulty_check, skill_check, combat_attack,
        get_player_state, get_npc_state,
        create_npc, invoke_npc, remove_npc, set_scene, game_over,
        search_knowledge, search_memory,
    ],
    model="deepseek-v4-flash",
)
```

**职责**：
- 分析玩家输入，判断是否需要检定
- 调用工具（骰子、检定、战斗、NPC）
- 综合所有结果进行叙事
- 提供建议行动选项

### NPC Agent

**定义**：[npc_agent.py](../trpg_agent/npc_agent.py)

```python
Agent(
    name=f"NPC_{name}",
    instructions=personality_prompt,  # 静态角色设定
    tools=[search_npc_memory],        # NPC 专用记忆搜索
    model="deepseek-v4-flash",
)
```

**生命周期**：

```
Game Start
  │
  ├─→ __init__()
  │     ├─→ _register_backstory_npcs()   # LLM 从角色背景提取 NPC → npc_store
  │     └─→ _sync_npc_agents()            # 遍历 npc_store，为缺失 NPC 创建 Agent
  │
  ├─→ load_save()
  │     ├─→ 恢复 scene_npcs 列表
  │     └─→ _sync_npc_agents()            # 为存档恢复的 NPC 创建 Agent
  │
  ├─→ create_npc 工具调用                  # 运行时动态创建 NPC
  │     ├─→ npc_store.create()
  │     ├─→ create_npc_agent() → 加入池
  │     └─→ 后续 invoke_npc 直接走 Phase 2 路径
  │
  ├─→ invoke_npc 工具调用                  # 路由到 NPC Agent (Phase 2)
  │     └─→ 若池中不存在 → Legacy fallback
  │
  └─→ remove_npc 工具调用                  # 从场景移除（Agent 保留在池中）
```

**关键**：`_sync_npc_agents()` 在初始化末和存档加载后自动运行，遍历 `npc_store.all()` 为尚未在池中的 NPC 创建 Agent。这样无论是背景故事 NPC、剧本 NPC 还是存档恢复的 NPC，首次 `invoke_npc` 就能走 Phase 2 路径，无需等待 `create_npc` 工具调用。

**自主能力**：
- 感知：读取场景上下文、玩家 prompt、自己的记忆
- 决策：根据性格、情绪、记忆决定回应方式
- 行动：说话、做动作、沉默/拒绝
- 记忆：通过 `search_npc_memory` 工具检索自己的过往记忆

---

## 工具系统

**文件**：[tools.py](../trpg_agent/tools.py)

全部 13 个工具通过 `@function_tool` 装饰器注册到 GM Agent。

### 判定工具

| 工具 | 签名 | 功能 |
|------|------|------|
| `roll_dice` | `(ctx, expression: str) -> str` | 通用骰子投掷，如 `"d20"`, `"3d6+2"` |
| `difficulty_check` | `(ctx, dc: int, modifier: int) -> str` | d20 ≥ DC 难度检定 |
| `skill_check` | `(ctx, skill_name: str, modifier: int) -> str` | d100 ≤ 技能值 技能检定 |
| `combat_attack` | `(ctx, target: str) -> str` | 攻击 NPC，d20≥DC12 命中 |

### 查询工具

| 工具 | 签名 | 功能 |
|------|------|------|
| `get_player_state` | `(ctx) -> str` | 查询玩家 HP/情绪/信任度/体力 |
| `get_npc_state` | `(ctx, name: str) -> str` | 查询 NPC 完整状态 |

### NPC 工具

| 工具 | 签名 | 功能 |
|------|------|------|
| `create_npc` | `(ctx, name, core, personality_tone) -> str` | 创建 NPC + 自动创建 NPC Agent |
| `invoke_npc` | `(ctx, name, prompt) -> str` | 路由到 NPC Agent（Phase 2）/ 旧逻辑（Phase 1 fallback） |
| `remove_npc` | `(ctx, name) -> str` | 从场景移除 NPC |
| `set_scene` | `(ctx, location, present_npcs, time_of_day, weather) -> str` | 更新场景信息 |

### 系统工具

| 工具 | 签名 | 功能 |
|------|------|------|
| `game_over` | `(ctx, cause) -> str` | 结束游戏，清理存档和记忆 |
| `search_knowledge` | `(ctx, query) -> str` | 搜索世界知识库 |
| `search_memory` | `(ctx, query) -> str` | 搜索冒险记忆 |

### invoke_npc 双路径

```
invoke_npc(ctx, name, prompt)
  │
  ├─→ NPC Agent 存在于池中? (Phase 2)
  │     ├─→ memory.npc_full_retrieve()   # NPC 记忆检索
  │     ├─→ build_npc_input()            # 动态上下文注入
  │     └─→ Runner.run_sync(npc_agent)   # NPC 独立推理
  │
  └─→ NPC Agent 不存在? (Phase 1 fallback)
        ├─→ npc.build_personality_prompt()
        ├─→ npc.build_state_prompt()
        ├─→ memory.npc_full_retrieve()
        └─→ LLM.chat(npc_system, npc_history)
```

---

## 记忆系统

### 存储架构

**文件**：[memory.py](../trpg_agent/memory.py)

```
MemoryStore
├── ChromaDB (向量语义搜索)
│   ├── "memories" 集合      — 主记忆 (event / fact / emotion_peak)
│   └── "npc_memories" 集合  — NPC 专属记忆 (npc_dialogue)
│
└── NetworkX DiGraph (关系图谱)
    ├── 节点：每条记忆 (id, content, type, importance, timestamp)
    └── 边关系：
        ├── "导致了"       — 因果关系
        ├── "关联到"       — 语义关联
        ├── "反驳了"       — 矛盾关系
        └── "发生在...之后" — 时间顺序
```

**嵌入模型**：`BAAI/bge-small-zh-v1.5` (SentenceTransformers)

### 写入流程

每个 `process()` 回合结束后触发：

```
玩家输入 + GM叙述 + 工具结果
  │
  ├─→ _write_memory()
  │     ├─→ LLM.extract_memory()     # 提取1-2句关键事件
  │     ├─→ memory.add()             # 写入 ChromaDB + 图谱节点
  │     ├─→ memory.link(关联到)      # 链接到语义相似记忆
  │     └─→ memory.link(发生在...之后) # 时间链
  │
  └─→ _write_npc_memory() (每个涉及的NPC)
        ├─→ LLM.extract_memory()     # 从NPC视角提取
        ├─→ memory.npc_add()         # 写入 npc_memories 集合
        └─→ memory.link(关联到)      # 交叉链接到主记忆
```

### 检索流程

**GM 记忆检索** — `_build_input()`：

```
用户输入
  → LLM.generate_search_query()   # 生成优化查询词 (20-50字)
  → memory.full_retrieve()        # 语义搜索 + 图遍历
      ├── ChromaDB 语义搜索 (top 5)
      ├── 每个结果 → 图遍历 2 跳
      ├── 过滤 npc_dialogue 类型
      └── 按 importance 降序排序
  → 合并 _recent_events (deque, maxlen=10)
  → 注入 "## 记忆" 段落
```

**NPC 记忆检索** — `invoke_npc`：

```
prompt + npc_name
  → memory.npc_full_retrieve()     # NPC专用集合语义搜索 + 图遍历
      ├── ChromaDB npc_memories 集合 (按 npc_name 过滤)
      ├── 每个结果 → 图遍历 2 跳 (仅匹配 npc_name 的 npc_dialogue 节点)
      └── 去重排序
  → 注入 NPC Agent input
```

---

## 上下文管理

### 回合内上下文

每次 `process()` 调用，`_build_input()` 组装完整的上下文字符串：

```
┌──────────────────────────────────────┐
│ ## 记忆                              │
│ - 上次在码头遇到商人... （关联到）     │
│ - 老马提到过黑猫的事                  │
│                                      │
│ ## 世界知识                          │
│ - 码头区最近不太平                    │
│ - 老马酒馆是信息集散地               │
│                                      │
│ ## 上一轮 GM 回应                    │
│ 老马擦了擦吧台...                    │
│ 1. 继续打听黑猫  2. 去码头  3. 离开  │
│                                      │
│ 玩家: 我想去酒馆找老马聊聊            │
│                                      │
│ 请在叙述结束时列出3个建议行动选项     │
└──────────────────────────────────────┘
```

### 上下文边界控制

| 组件 | 容量 | 裁剪策略 |
|------|------|----------|
| `history_messages` | 最近 5 轮 (10 条消息) | `_trim_history()` 截断 |
| `_recent_events` | 最近 10 条事件 | `deque(maxlen=10)` 自动丢弃 |
| `_last_gm_response` | 1 轮 | 每次覆盖 |
| NPC 对话历史 (`NPCStore._histories`) | 每 NPC 最近 10 轮 (20 条消息) | 滑动窗口截断 |
| 记忆检索结果 | 语义 top 5 + 图谱扩展 + recent 5 → 最多 8 条 | `seen_prefixes` 去重 |
| 知识检索结果 | top 3 | 截断 |

### 跨回合状态持久化

```
GameMaster 实例 (内存)
  ├── player_state (StateMachine)     → 每次都变
  ├── scene_npcs                      → 通过 set_scene / remove_npc 变更
  ├── _time_of_day / _weather         → 通过 set_scene 变更
  ├── _game_over                      → 通过 game_over 变更
  ├── history_messages                → 每次 process() 后更新
  └── _recent_events                  → 每次 _write_memory() 后追加

save.json (磁盘)
  ├── time_of_day, weather
  ├── player_state (to_dict)
  ├── scene_npcs
  ├── recent_events
  ├── last_gm_response
  └── turn_count

ChromaDB (磁盘)
  ├── memories/     → 主记忆集合 (持久化向量 + 元数据)
  ├── npcs/         → NPC 角色卡集合
  └── knowledge/    → 世界知识集合
```

---

## 数据流

### 单回合完整流程

```
Player Input: "老马，来杯麦酒！"
  │
  ├─ 1. GameMaster.process()
  │     ├── _build_input(user_input)
  │     │     ├── LLM.generate_search_query()
  │     │     ├── MemoryStore.full_retrieve()
  │     │     ├── KnowledgeBase.query()
  │     │     └── 组装 input 文本
  │     │
  │     └── Runner.run_sync(gm_agent, input, context=game_ctx)
  │           │
  │           ├── [SDK Turn 1] GM: create_npc(name="老马", core="...", personality_tone="粗豪")
  │           │     └── 工具执行: NPCStore.create() + create_npc_agent() → 加入池
  │           │
  │           ├── [SDK Turn 2] GM: invoke_npc(name="老马", prompt="玩家向你搭话，要一杯麦酒")
  │           │     └── 工具执行:
  │           │           ├── npc_agents["老马"] 存在 → Phase 2 路径
  │           │           ├── MemoryStore.npc_full_retrieve("老马", prompt)
  │           │           ├── build_npc_input(prompt, context, history)
  │           │           ├── Runner.run_sync(npc_agent, npc_input)
  │           │           │     └── NPC Agent 推理: "来了来了，这就给您满上！"
  │           │           └── 返回: 老马: "来了来了，这就给您满上！"
  │           │
  │           └── [SDK Turn 3] GM: 最终叙述 (无更多 tool calls)
  │                 └── 输出: "老马转身从架子上取下一只杯子...\n\n1. 打听黑猫的事\n2. 观察酒馆里的其他客人\n3. 喝完酒离开"
  │
  ├─ 2. 后处理
  │     ├── history_messages.append(user_input)
  │     ├── history_messages.append(response)
  │     ├── _trim_history()
  │     ├── _write_memory()
  │     ├── _write_npc_memory("老马", ...)
  │     └── _maybe_trigger_world_simulation()
  │
  └─ 3. return response
```

---

## 文件索引

| 文件 | 职责 |
|------|------|
| [trpg_agent/agent_config.py](../trpg_agent/agent_config.py) | SDK 初始化、`GameContext` dataclass、GM system prompt、`create_gm_agent()` |
| [trpg_agent/tools.py](../trpg_agent/tools.py) | 全部 13 个 `@function_tool` + `invoke_npc` 双路径路由 |
| [trpg_agent/npc_agent.py](../trpg_agent/npc_agent.py) | `create_npc_agent()` 工厂、`build_npc_input()`、NPC 记忆搜索工具 |
| [trpg_agent/game_master.py](../trpg_agent/game_master.py) | `GameMaster` 主控类、`process()`、记忆写入、世界模拟触发 |
| [trpg_agent/memory.py](../trpg_agent/memory.py) | `MemoryStore` — ChromaDB + NetworkX 混合存储 |
| [trpg_agent/llm.py](../trpg_agent/llm.py) | DeepSeek API 封装（已被 SDK 逐步替代，保留 extract_memory 等方法） |
| [trpg_agent/npc.py](../trpg_agent/npc.py) | `NPCCharacter` 模型 + `NPCStore` 持久化管理 |
| [trpg_agent/state.py](../trpg_agent/state.py) | `StateMachine` — HP/情绪/信任度/体力 状态机 |
| [trpg_agent/world_builder.py](../trpg_agent/world_builder.py) | `WorldBuilder` — LLM 驱动的 NPC/世界创建 |
| [trpg_agent/api_server.py](../trpg_agent/api_server.py) | FastAPI 服务、SSE 流式输出 |
