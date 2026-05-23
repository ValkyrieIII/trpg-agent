# TRPG Agent

AI 驱动的单人跑团应用。玩家扮演角色冒险，AI 担任 GM（叙事/检定）+ NPC 扮演。

## 架构

```
玩家输入 → GameMaster.process()
              │
              ▼
         GM Agent (LLM)
              │ 每轮返回 JSON:
              │ {thinking, narration, tool_calls, involved_npcs, persist_memory, suggestions}
              │
              ├── 纯叙述 → 渲染输出
              └── tool_calls → 执行工具 → 结果回传 → 继续循环
                   │
                   ├── roll_dice / difficulty_check / skill_check   (检定)
                   ├── combat_attack / madness_check                (战斗/疯狂)
                   ├── get_player_state / get_npc_state             (查询)
                   ├── create_npc / npc_speak / remove_npc / set_scene (NPC)
                   ├── search_knowledge / search_memory              (检索)
                   └── game_over                                     (终结)
```

## 记忆系统

```
主记忆库 (ChromaDB collection: memories)
  ├── 语义检索 + NetworkX 图遍历
  ├── 边关系: "关联到" / "发生在...之后"
  └── 只存玩家事件，不含 NPC 记忆

NPC 记忆库 (ChromaDB collection: npc_memories, 按 npc_name 过滤)
  ├── 语义检索 + 图遍历 (只查该 NPC 自己的记忆)
  └── 独立存储，NPC 间记忆隔离
```

## 目录

```
trpg_agent/
├── main.py          CLI 入口 (Rich)
├── game_master.py   GM 核心（工具调用循环 + prompt 管理）
├── character.py     玩家角色卡 (YAML → dataclass)
├── npc.py           NPC 角色卡 + ChromaDB 持久化 + 状态JSON
├── state.py         状态机 (HP/情绪/信任/体力/疯狂)
├── dice.py          骰子 (d20/d100/ndm+k)
├── check.py         检定 (技能/难度/对抗)
├── event.py         事件判定 (已废弃，GM Agent 替代)
├── llm.py           LLM 封装 (DeepSeek, OpenAI SDK, JSON 强制输出)
├── memory.py        记忆 (双 ChromaDB collection + NetworkX 图)
└── rag.py           知识库 (ChromaDB)
config.yaml          玩家角色 & 世界设定 & 人物关系
data/knowledge/      世界观知识 & 剧本 (.md)
data/save.json       游戏存档
```

## 运行

```bash
# 1. 设环境变量
set DEEPSEEK_API_KEY=sk-xxx
set HF_ENDPOINT=https://hf-mirror.com

# 2. 启动
python trpg_agent/main.py --debug
```

## 功能

| 功能 | 说明 |
|------|------|
| 断点续存 | 退出/Ctrl+C 自动保存 `data/save.json`，重启续接 |
| 角色死亡 | HP≤0 / 疯狂≥100 / 叙事死亡 → 清档重启 |
| 剧本系统 | `data/knowledge/scenario_black_cat.md` 节点式剧本 |
| NPC 关系 | `config.yaml` 的 `relations` + NPC 创建时的 `relations` |
| NPC 记忆 | 独立向量库，按 NPC 过滤，LLM 提取摘要 |
| 场景管理 | set_scene/remove_npc 维护场景 NPC 列表 |

## 测试

```bash
pytest tests/ -v
```
