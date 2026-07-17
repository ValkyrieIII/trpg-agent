# TRPG Agent — 项目状态总览

> 生成日期：2026-07-17 | 基于最新提交 `f3aae63`

---

## 1. 项目简介

**TRPG Agent** 是一个 AI 驱动的单人跑团应用。玩家扮演冒险者，AI 担任 GM（叙事/检定/NPC 扮演）。项目基于 **OpenAI Agents SDK** 编排多个 AI Agent，支持实时流式输出和多 NPC 独立推理。

## 2. 目录结构

```
📁 trpg_agent/              ← Python 后端（核心包）
├── main.py                  CLI 入口 (Rich 终端界面)
├── game_master.py           GM 核心 (102KB，约 2100 行，最大的文件)
├── agent_config.py          SDK 初始化 + GameContext + GM system prompt
├── tools.py                 13 个 @function_tool（检定/战斗/NPC/场景/搜索）
├── npc_agent.py             NPC Agent 工厂 + build_npc_input()
├── memory.py                双 ChromaDB 向量库 + NetworkX 记忆图谱
├── character.py             玩家角色卡（YAML → dataclass）
├── npc.py                   NPC 角色卡 + ChromaDB 持久化 + 状态 JSON
├── dice.py                  骰子系统（d20/d100/ndm+k）
├── check.py                 检定引擎（技能/难度）
├── rag.py                   知识库（ChromaDB + 角色权限过滤）
├── state.py                 状态机（HP/情绪/信任/体力）
├── world_builder.py         LLM 驱动的世界构建器
├── llm.py                   DeepSeek API 封装（逐步被 SDK 替代）
├── api_server.py            FastAPI 后端 + SSE 流式输出
└── event_stream.py          Agent 状态事件流（前端实时展示）

📁 web/                      ← TypeScript 前端 (React + Vite)
├── src/
│   ├── App.tsx              根组件：StartScreen 或 GameInterface
│   ├── main.tsx             React 入口
│   ├── index.css            全局样式（暗色主题）
│   ├── components/
│   │   ├── StartScreen.tsx          开始界面
│   │   ├── GameInterface.tsx        游戏主界面
│   │   ├── StatePanel.tsx           角色状态面板
│   │   ├── ModalNPC.tsx             NPC 详情弹窗
│   │   ├── ModalKnowledge.tsx       知识库浏览弹窗
│   │   └── ModalSettings.tsx        设置弹窗
│   └── store/
│       └── gameStore.ts     Zustand 状态管理
├── tailwind.config.js       Tailwind CSS 配置
└── vite.config.ts           Vite 配置（/api 代理到 :8000）

📁 tests/                    ← 测试（8 个文件，约 119 个用例）
├── test_dice.py             test_character.py
├── test_check.py            test_event.py
├── test_memory.py           test_npc.py
├── test_rag.py              test_state.py

📁 docs/                     ← 文档
├── ARCHITECTURE.md          旧架构文档
├── MULTI_AGENT_ARCHITECTURE.md  当前多 Agent 架构设计
├── GAMEPLAY.md              玩家指南
├── plans/                   设计文档（4 份）
└── PROJECT_STATUS.md        本文件

📁 data/                     ← 游戏数据
├── chroma/                  ChromaDB 持久化向量库
├── knowledge/               世界观知识文件 (.md)
├── npcs/                    NPC 角色卡 (.yaml)
├── memory_graph.json        NetworkX 图谱导出
└── save.json                游戏存档

📁 参考/                     参考材料
📁 实际测试/                 测试日志
📁 世界生成/                 世界生成输出
```

## 3. 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **LLM** | DeepSeek V4 | 通过 OpenAI SDK Chat Completions 兼容模式调用 |
| **Agent 框架** | openai-agents ≥ 0.17.0 | 多 Agent 编排，Runner.run_sync() 自动管理工具调用循环 |
| **后端** | Python 3.11+ | FastAPI + uvicorn |
| **前端** | React 18 + TypeScript 5.5 | Vite 5 构建，Zustand 4 状态管理 |
| **样式** | Tailwind CSS 3 | 暗色主题 |
| **向量库** | ChromaDB ≥ 0.4.0 | 双 collection：memories + npc_memories |
| **图数据库** | NetworkX ≥ 3.0 | 记忆关系图谱，支持 2 跳遍历 |
| **嵌入模型** | BAAI/bge-small-zh-v1.5 | ~24MB |
| **骰子系统** | 自研 d20/d100/ndm+k | 支持加减修正 |
| **终端 UI** | Rich ≥ 13.0 | CLI 模式面板、spinner、颜色 |

## 4. 架构概览

### 总体架构：客户端/服务器 + CLI

```
┌──────────────────────────────────────────────────────────────┐
│                      客户端层                                │
│  ┌──────────────┐      ┌─────────────────────────────────┐  │
│  │ CLI (Rich)   │      │ Web 前端 (React + Vite)         │  │
│  │ main.py      │      │ localhost:5173 (dev)             │  │
│  │              │      │ /api 代理 → localhost:8000      │  │
│  └──────┬───────┘      └───────────────┬─────────────────┘  │
└─────────┼─────────────────────────────┼─────────────────────┘
          │                             │ HTTP REST + SSE
          ▼                             ▼
┌──────────────────────────────────────────────────────────────┐
│                      服务端层 (Python)                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │ FastAPI Server (api_server.py)  Port 8000         │     │
│  └────────────────────┬───────────────────────────────┘     │
│                       ▼                                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │              GameMaster (game_master.py)            │     │
│  │                                                      │     │
│  │  process() / process_streaming()                     │     │
│  │    ├─ _build_input() → 记忆检索 + 知识查询          │     │
│  │    ├─ Runner.run_sync(gm_agent, ...)                │     │
│  │    │     ├─ GM Agent（编排者）                       │     │
│  │    │     │   ├─ 13 个 @function_tool 工具           │     │
│  │    │     │   └─ invoke_npc → NPC Agent 池           │     │
│  │    │     └─ NPC Agents（独立推理）                   │     │
│  │    ├─ _write_memory() / _write_npc_memory()         │     │
│  │    └─ _maybe_trigger_world_simulation()             │     │
│  └────────────────────────────────────────────────────┘     │
│                       │                                       │
│         ┌─────────────┼──────────────┐                       │
│         ▼             ▼              ▼                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ ChromaDB │  │ NetworkX │  │ 角色/NPC │                   │
│  │ 3 colls  │  │ 记忆图谱 │  │ 状态机   │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

### Player Input → GM 处理流程

```
玩家输入
  │
  ▼
GameMaster.process()
  │
  ├─→ _build_input()                     # 记忆检索 + 知识查询 + 历史拼装
  │
  ├─→ Runner.run_sync(gm_agent, ...)     # SDK 自动管理工具调用循环
  │     │
  │     ├─→ GM Agent (编排者)
  │     │     ├── roll_dice / difficulty_check / skill_check   # 检定
  │     │     ├── combat_attack                                # 战斗
  │     │     ├── get_player_state / get_npc_state             # 状态查询
  │     │     ├── create_npc / remove_npc / set_scene          # 场景管理
  │     │     ├── invoke_npc ──→ NPC Agent (独立推理)          # NPC
  │     │     ├── search_knowledge / search_memory              # 检索
  │     │     └── game_over                                     # 终结
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

### 架构设计要点

1. **编排者模式（非 Handoff）**：GM Agent 始终保持控制权，NPC Agent 交互通过 `invoke_npc` 工具调用触发。这支持多 NPC 群聊和统一叙事。

2. **OpenAI Agents SDK 托管循环**：SDK 通过 `Runner.run_sync()` 自动管理工具调用循环，替代了旧架构中手动 while 循环 + JSON `tool_calls` 解析。

3. **GameContext 依赖注入**：所有 13 个工具通过 `RunContextWrapper[GameContext]` 获取状态，与 GameMaster 类解耦。

4. **双 ChromaDB 记忆系统**：主记忆库（玩家事件，跨 NPC 关联）与 NPC 记忆库（按 NPC 名称隔离），结合 NetworkX 图谱的边关系（"导致了"/"关联到"/"反驳了"/"发生在...之后"）进行 2 跳遍历。

5. **统一 NPC 入口**：单个 `invoke_npc(name, prompt)` 工具动态路由到对应 NPC Agent，避免为每个 NPC 硬编码工具。

6. **实时 SSE 流式**：Web 前端通过 `AgentStatusHooks` + Server-Sent Events 实时接收 Agent 状态更新（思考、工具调用参数/结果、NPC 对话等）。

7. **增量演进**：项目最初是三层手工管线（正则→嵌入→LLM），后重写为基于 SDK 的多 Agent 架构。旧代码以注释形式保留。

## 5. 记忆系统

```
主记忆库 (ChromaDB collection: memories)
  ├── 语义检索 + NetworkX 图遍历 (2跳)
  ├── 边关系: "导致了" / "关联到" / "反驳了" / "发生在...之后"
  └── 只存玩家事件，不含 NPC 记忆

NPC 记忆库 (ChromaDB collection: npc_memories, 按 npc_name 过滤)
  ├── 语义检索 + 图遍历 (只查该 NPC 自己的记忆)
  └── 独立存储，NPC 间记忆隔离
```

### 上下文管理策略

| 组件 | 容量 | 策略 |
|------|------|------|
| `history_messages` | 最近 5 轮 | `_trim_history()` 截断 |
| `_recent_events` | 最近 10 条 | `deque(maxlen=10)` |
| 记忆检索结果 | 最多 8 条 | 语义 top 5 + 图扩展 + 去重 |
| NPC 对话历史 | 每 NPC 10 轮 | `NPCStore` 滑动窗口 |

## 6. 功能清单

| 功能 | 说明 | 状态 |
|------|------|------|
| 多 Agent NPC | 每个 NPC 具有独立 Agent，自主决策回应方式和内容 | ✅ |
| 断点续存 | 退出/Ctrl+C 自动保存 `data/save.json`，重启续接 | ✅ |
| 角色死亡 | HP≤0 / 叙事死亡 → game_over 清档重启 | ✅ |
| 剧本系统 | `data/knowledge/scenario_black_cat.md` 节点式剧本 | ✅ |
| NPC 关系 | `config.yaml` 的 `relations` + NPC 创建时的 `relations` | ✅ |
| NPC 记忆 | 独立向量库，按 NPC 过滤，LLM 提取摘要 | ✅ |
| 场景管理 | set_scene / remove_npc 维护场景 NPC 列表 | ✅ |
| 世界模拟 | 每 N 回合 NPC 自主行动，GM 叙事包装 | ✅ |
| Web 前端 | React 前端，实时展示 LLM 思考/工具调用状态 | ✅ |
| CLI 模式 | Rich 终端 UI，支持 `/dice` 等斜杠命令 | ✅ |
| 调试模式 | `--debug` 标志输出每轮管线诊断信息 | ✅ |
| 掷骰 | d20/d100/ndm+k，支持加减修正 | ✅ |
| 技能检定 | 技能值/难度检定/对抗检定 | ✅ |
| 知识库 | ChromaDB 世界观知识，角色权限过滤 | ✅ |
| 存档相容 | `data/save.json` JSON 格式，支持断点续玩 | ✅ |

## 7. 运行方式

### 前置条件

```bash
# .env 文件配置
DEEPSEEK_API_KEY=sk-xxx
```

### CLI 模式

```bash
# 激活虚拟环境后启动
.venv\Scripts\python.exe -m trpg_agent.main --debug
```

### Web 前后端模式

```bash
# 一键启动（PowerShell）
.\dev.ps1

# 前端 → http://localhost:5173
# 后端 → http://localhost:8000
```

### 运行测试

```bash
pytest tests/ -v
```

### CLI 命令

| 命令 | 说明 |
|------|------|
| `/dice 3d6` | 投掷 3 个 6 面骰 |
| `/clear_npc <名称>` | 清除指定 NPC 对话记忆 |
| `/clear_npc all` | 清除全部 NPC 对话记忆 |
| `exit` / `quit` / `退出` | 保存并退出 |

## 8. Git 历史摘要

```
f3aae63 Create memory_graph.json                          ← 最新 (5月26日)
06a8f77 feat: 部署配置修复 + GM输出格式约束 + NPC群聊 + UI改进
e4c6145 (简略提交)
a3773b2 feat: 前端实时展示LLM思考/工具调用状态 + JSON格式分离叙事与建议
8bff5b2 换用openai sdk架构                                 ← 重大重写
f8e9e66 Merge PR #1: 前端优化 + 后端
ca1a161 前端优化加上后端
c1d3d7f 前端（初始）
c6aa8e2 feat: GM Agent 工具调用架构重写
4360169 feat: unified post-process pipeline for all handlers
6623afe docs: add architecture overview for developer onboarding
cadfa6d feat: unified StateMachine with HP for players and NPCs
7877cd3 feat: implement three-tier action classification pipeline
8ce099b fix: route 射击 to combat event, enforce short NPC names
40c105c feat: add GM opening scene narration on game start
9db6bf0 feat: show dice roll results in all event check narratives
ea8cb67 feat: add /clear_npc command to reset NPC conversation history
293a0bd feat: add --debug flag for per-turn pipeline diagnostics
fb615c3 feat: switch embedding model to BAAI/bge-small-zh-v1.5
ee041df feat: add embedding-based event routing + combat trigger
```

### 远程同步状态（2026-07-17）

| 分支 | 状态 |
|------|------|
| 本地 `master` ↔ `origin/master` | ✅ 完全同步 |
| `origin/main` | ⚠️ 停在 `06a8f77`，落后 1 个提交 |

## 9. 改造历程

| 维度 | 旧架构 | 新架构 |
|------|--------|--------|
| 工具调用 | GM 输出 JSON `tool_calls`，手动分发 | SDK native function calling，自动循环 |
| 循环管理 | 手动 while 循环 + messages 拼接 | `Runner.run_sync()` 自动管理 |
| NPC 回应 | `_tool_npc_speak` → `LLM.chat()` 被动生成 | NPC Agent 独立 LLM 推理，自主决策 |
| 世界模拟 | 硬编码每 3 回合触发 | 保留外部触发，内部可用 NPC Agent |
| 状态注入 | `self.xxx` 直接访问 GameMaster | `RunContextWrapper[GameContext]` 统一注入 |

## 10. 当前状态备注

- ⏱️ 上次提交：2026-05-26，距今约 7 周
- 🧹 工作区干净，无未提交更改
- 🐍 虚拟环境 `.venv` 完整，依赖已安装
- 🧪 8 个测试文件，覆盖骰子、角色、状态、检定、事件、记忆、NPC、RAG
- 📦 `game_master.py` 是最大的文件（约 2100 行），承载 GM 核心逻辑
- 🔑 `.env` 中的 API Key 需确认是否仍有效
