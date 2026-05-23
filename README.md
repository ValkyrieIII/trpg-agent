# TRPG Agent

单双 Agent 跑团应用。GM 负责叙述和裁定，角色负责对话。

## 架构

```
玩家输入 → GM Agent(场景叙述+检定) → 角色 Agent(对话)
              ↑                           ↑
         system: "你是DM"           system: 角色卡(config.yaml)
```

## 目录

```
trpg_agent/
├── main.py          CLI 入口 (Rich)
├── game_master.py   GM 核心 (双 Agent 管线)
├── character.py     角色卡 (YAML → prompt)
├── state.py         状态机 (情绪/信任/体力)
├── dice.py          骰子 (d20/d100/ndm+k)
├── check.py         检定 (技能/难度/对抗)
├── event.py         事件判定 (陷阱/NPC/发现)
├── llm.py           LLM 封装 (DeepSeek, OpenAI SDK)
├── memory.py        记忆 (ChromaDB + NetworkX 图)
└── rag.py           知识库 (角色权限过滤)
config.yaml          角色 & 知识配置
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

## 记忆系统

- **GM 叙述不写入当前记忆** — 仅角色对话进入 MemoryStore
- GM 会重复场景叙述的风险：TODO: 拆分为世界记忆(GM) + 角色记忆(NPC)

## 测试

```bash
pytest tests/test_dice.py tests/test_character.py tests/test_state.py tests/test_check.py tests/test_event.py -v
```
