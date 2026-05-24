# TRPG Agent

AI 驱动的单人跑团应用。玩家扮演角色冒险，AI 担任 GM（叙事/检定）+ NPC 扮演。

## 架构

基于 **OpenAI Agents SDK** 的多 Agent 编排系统。

```
Player Input
  │
  ▼
GameMaster.process()
  │
  ├─→ _build_input()                     # 记忆检索 + 知识查询 + 历史拼装
  │
  ├─→ Runner.run_sync(gm_agent, ...)     # SDK 自动管理工具调用循环
  │     │
  │     ├─→ GM Agent (编排者)
  │     │     ├── roll_dice / difficulty_check / skill_check   (检定)
  │     │     ├── combat_attack                                (战斗)
  │     │     ├── get_player_state / get_npc_state             (查询)
  │     │     ├── create_npc / remove_npc / set_scene           (场景)
  │     │     ├── invoke_npc ──→ NPC Agent (独立推理)           (NPC)
  │     │     ├── search_knowledge / search_memory              (检索)
  │     │     └── game_over                                     (终结)
  │     │
  │     └─→ NPC Agent Pool (每个 NPC 一个独立 Agent)
  │           ├── 静态 instructions: 性格、说话风格、口头禅
  │           ├── 动态 input: 情绪、体力、场景、相关记忆
  │           └── 工具: search_npc_memory
  │
  ├─→ _write_memory() / _write_npc_memory()   # 记忆持久化
  ├─→ _maybe_trigger_world_simulation()       # 世界模拟（定时触发）
  └─→ 返回 narration + suggestions
```

### 改造前 vs 改造后

| 维度 | 旧架构 | 新架构 |
|------|--------|--------|
| 工具调用 | GM 输出 JSON `tool_calls`，手动分发 | SDK native function calling，自动循环 |
| 循环管理 | 手动 while 循环 + messages 拼接 | `Runner.run_sync()` 自动管理 |
| NPC 回应 | `_tool_npc_speak` → `LLM.chat()` 被动生成 | NPC Agent 独立 LLM 推理，自主决策 |
| 世界模拟 | 硬编码每 3 回合触发 | 保留外部触发，内部可用 NPC Agent |
| 状态注入 | `self.xxx` 直接访问 GameMaster | `RunContextWrapper[GameContext]` 统一注入 |

## 记忆系统

```
主记忆库 (ChromaDB collection: memories)
  ├── 语义检索 + NetworkX 图遍历 (2跳)
  ├── 边关系: "导致了" / "关联到" / "反驳了" / "发生在...之后"
  └── 只存玩家事件，不含 NPC 记忆

NPC 记忆库 (ChromaDB collection: npc_memories, 按 npc_name 过滤)
  ├── 语义检索 + 图遍历 (只查该 NPC 自己的记忆)
  └── 独立存储，NPC 间记忆隔离
```

### 上下文管理

| 组件 | 容量 | 策略 |
|------|------|------|
| `history_messages` | 最近 5 轮 | `_trim_history()` 截断 |
| `_recent_events` | 最近 10 条 | `deque(maxlen=10)` |
| 记忆检索结果 | 最多 8 条 | 语义 top 5 + 图扩展 + 去重 |
| NPC 对话历史 | 每 NPC 10 轮 | `NPCStore` 滑动窗口 |

## 目录

```
trpg_agent/
├── main.py           CLI 入口 (Rich)
├── game_master.py    GM 核心 (Runner.run_sync + 记忆写入 + 世界模拟)
├── agent_config.py   SDK 初始化、GameContext、GM system prompt
├── tools.py          全部 13 个 @function_tool + invoke_npc 双路径路由
├── npc_agent.py      NPC Agent 工厂、build_npc_input()
├── character.py      玩家角色卡 (YAML → dataclass)
├── npc.py            NPC 角色卡 + ChromaDB 持久化 + 状态JSON
├── state.py          状态机 (HP/情绪/信任/体力)
├── dice.py           骰子 (d20/d100/ndm+k)
├── check.py          检定 (技能/难度)
├── llm.py            LLM 封装 (DeepSeek, 逐步被 SDK 替代)
├── memory.py         记忆 (双 ChromaDB collection + NetworkX 图)
├── world_builder.py  世界构建器 (LLM 驱动的 NPC/知识创建)
├── rag.py            知识库 (ChromaDB)
└── api_server.py     FastAPI + SSE 流式输出
config.yaml           玩家角色 & 世界设定
data/save.json        游戏存档
data/chroma/          ChromaDB 持久化 (memories / npcs / knowledge)
```

## 运行

```bash
# 1. 配置 API Key (.env 文件)
DEEPSEEK_API_KEY=sk-xxx

# 2. CLI 模式
.venv\Scripts\python.exe -m trpg_agent.main --debug

# 3. Web 模式 (前后端)
.\dev.ps1
# 前端: http://localhost:5173
# 后端: http://localhost:8000
```

> **DeepSeek 兼容性**：SDK 默认使用 OpenAI Responses API，DeepSeek 不支持。已通过 `set_default_openai_api("chat_completions")` 强制切换。

## 功能

| 功能 | 说明 |
|------|------|
| 断点续存 | 退出/Ctrl+C 自动保存 `data/save.json`，重启续接 |
| 多 Agent NPC | 每个 NPC 具有独立 Agent，自主决策回应方式和内容 |
| 角色死亡 | HP≤0 / 叙事死亡 → game_over 清档重启 |
| 剧本系统 | `data/knowledge/scenario_black_cat.md` 节点式剧本 |
| NPC 关系 | `config.yaml` 的 `relations` + NPC 创建时的 `relations` |
| NPC 记忆 | 独立向量库，按 NPC 过滤，LLM 提取摘要 |
| 场景管理 | set_scene / remove_npc 维护场景 NPC 列表 |
| 世界模拟 | 每 N 回合 NPC 自主行动，GM 叙事包装 |

## 文档

- [多 Agent 编排架构](docs/MULTI_AGENT_ARCHITECTURE.md) — 完整设计文档
