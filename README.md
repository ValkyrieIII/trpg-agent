# TRPG Agent

AI 驱动的单人跑团应用。玩家扮演角色冒险，AI 担任 GM（叙事/检定）+ NPC 扮演。

## 架构

```
玩家（扮演角色）
    │
   输入（行动/对话）
    │
    ▼
GM Agent（叙事中枢 + 检定裁判）
    │
    ├── 场景叙述（第三人称）
    ├── 检定判定与执行
    └── NPC 回应 → 独立 NPC Agent（角色卡 system prompt）
```

## 目录

```
trpg_agent/
├── main.py          CLI 入口 (Rich)
├── game_master.py   GM 核心（调度中枢）
├── character.py     玩家角色卡 (YAML → dataclass)
├── npc.py           NPC 角色卡 + ChromaDB 持久化
├── state.py         状态机 (情绪/信任/体力)
├── dice.py          骰子 (d20/d100/ndm+k)
├── check.py         检定 (技能/难度/对抗)
├── event.py         事件判定 (陷阱/NPC/发现)
├── llm.py           LLM 封装 (DeepSeek, OpenAI SDK)
├── memory.py        记忆 (ChromaDB + NetworkX 图)
└── rag.py           知识库 (角色权限过滤)
config.yaml          玩家角色 & 世界设定
data/npcs/           NPC 角色卡 (.yaml)
data/knowledge/      世界观知识文件 (.md)
```

## 运行

```bash
# 1. 创建 .env
echo DEEPSEEK_API_KEY=sk-xxx >> .env
echo HF_ENDPOINT=https://hf-mirror.com >> .env

# 2. 启动
python trpg_agent/main.py
```

## 测试

```bash
pytest tests/ -v
```
