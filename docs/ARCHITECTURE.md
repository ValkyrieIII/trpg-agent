# TRPG Agent 架构文档

## 概述

TRPG Agent 是一个 AI 驱动的单人跑团应用。玩家扮演角色冒险，AI 担任 GM（叙事/检定）和 NPC 扮演。

- **语言：** Python 3.13+
- **LLM：** DeepSeek V4 (OpenAI SDK 兼容)
- **向量库：** ChromaDB (持久化)
- **embedding：** BAAI/bge-small-zh-v1.5 (24MB, 中文优化)
- **图关系：** NetworkX (记忆关联)
- **CLI：** Rich

安装依赖：`pip install -r requirements.txt`

---

## 目录结构

```
trpg_agent/
├── main.py          CLI 入口 (Rich)，命令路由
├── game_master.py   GM 调度中枢，三层判定管线，双 Agent 调度
├── character.py     玩家角色卡 (YAML → dataclass)
├── npc.py           NPC 角色卡 + ChromaDB 持久化 + 状态管理
├── state.py         统一状态机 (HP/情绪/信任/体力)
├── dice.py          骰子表达式解析 + 投掷
├── check.py         检定引擎 (技能/难度/对抗)
├── event.py         事件判定 (陷阱/环境/NPC反应/发现/战斗)
├── llm.py           DeepSeek API 封装 (OpenAI SDK)
├── memory.py        混合记忆 (ChromaDB 语义 + NetworkX 关系图)
└── rag.py           知识库 (角色权限过滤)

config.yaml          玩家角色 & 世界设定
data/npcs/           NPC 角色卡 (.yaml)
data/knowledge/      世界观知识文件 (.md)
data/chroma/         ChromaDB 持久化数据 (自动生成)
```

---

## 架构图

```
                                 main.py (Rich CLI)
                                      │
                              ┌───────┴───────┐
                              │  GameMaster    │
                              │  调度中枢       │
                              └───────┬───────┘
                                      │
              player ─────────────────┼────────────────── world
           (config.yaml)              │              (config.yaml)
                                      │
    ┌─────────────┬─────────────┬─────┴─────┬─────────────┬─────────────┐
    │             │             │           │             │             │
    ▼             ▼             ▼           ▼             ▼             ▼
 character    npc_store     player_      memory       knowledge      llm
 (属性/技能)   (NPCStore)    state        (MemoryStore) (KnowledgeBase) (OpenAI SDK)
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
                state.py   dice.py   event.py
               (StateMachine) (roll)(resolve_trigger)
                    │         │         │
                    └─────────┼─────────┘
                              │
                              ▼
                          check.py
                    (skill/difficulty/opposed)
```

---

## 管线流程：三轮判定

```
玩家输入
    │
    ▼
process()
    │
    ├── Tier 1: 正则判定 (_detect_intent)
    │     ├── dice   → _handle_dice   → dice.roll()
    │     ├── info   → _handle_info   → player.summary() + player_state
    │     ├── event  → _handle_event  → embedding分类 → resolve_trigger()
    │     └── dialogue → 
    │                     │
    │                     ├── Tier 2: embedding 判定 (_classify_event)
    │                     │     匹配 combat/trap/environment/discovery/npc_reaction
    │                     │     → _handle_event → resolve_trigger()
    │                     │
    │                     └── Tier 3: LLM 判定 (_handle_dialogue)
    │                           │
    │                 ┌─────────┼─────────┐
    │                 ▼                   ▼
    │           _call_gm (LLM)     _match_action (正则)
    │           GM 分析输入          硬编码行动规则
    │           返回 JSON            返回检定参数
    │                 │                   │
    │                 └─────────┬─────────┘
    │                           │
    │               ┌───────────┼───────────┐
    │               ▼           ▼           ▼
    │           GM 叙事      检定执行      NPC Agent
    │           (第三人称)  (_execute_check) (独立LLM调用)
    │
    └── 状态更新 → 记忆写入 → player_history 滑动窗口
```

### 正则判定 (Tier 1)

```python
_PATTERN_DICE  = r"掷骰|骰子|d\d+"          # → dice
_PATTERN_INFO  = r"查看|属性|状态|角色卡"     # → info
_PATTERN_EVENT = r"战斗|攻击|射击|触发|陷阱|环境"  # → event
```

### Embedding 判定 (Tier 2)

五类事件的中心向量（启动时预计算）：

```python
_EVENT_DESCRIPTIONS = {
    "combat":        "战斗 攻击 射击 砍杀 挥拳 ...",
    "trap":          "陷阱 机关 暗器 ...",
    "environment":   "环境 天气 地形 攀爬 ...",
    "discovery":     "发现 线索 搜索 观察 ...",
    "npc_reaction":  "NPC反应 对话 交涉 说服 ...",
}
```

用户输入向量 → cosine 相似度 vs 5 个中心向量 → 最高分 ≥0.4 即匹配

### LLM 判定 (Tier 3)

GM Agent system prompt 注入：场景 NPC 列表、玩家角色卡、检定规则、知识/记忆。要求返回 JSON：

```json
{"narration": "...", "check": null, "responding_npc": null, "new_npc": null}
```

JSON 解析失败则降级为全文当作 narration 处理。

---

## 组件说明

### character.py — 玩家角色卡

```python
@dataclass
class Character:
    name: str                    # 角色名
    core: List[str]              # 背景描述
    attributes: Dict[str, int]   # 属性
    skills: List[Dict]           # 技能 [{name, value}]
```

从 `config.yaml` 的 `player:` key 加载。不含 personality/few_shot（归 NPC）。

### npc.py — NPC 角色卡 + 持久化

```python
@dataclass
class NPCCharacter(Character):
    personality: Dict   # {tone, verbal_tics, emotion_map, catchphrases}
    few_shot: List[Dict]  # [{input, output}]
```

```python
class NPCStore:
    # 方法
    save(npc)           # 持久化到 ChromaDB
    find_by_name(name)  # 精确查找
    search(query)       # 语义搜索
    all()               # 全部 NPC
    create(name, ...)   # 动态创建 (GM 在叙事中引入新 NPC)
    get_state(name)     # 获取 NPC 的 StateMachine
    get_history(name)   # NPC 对话历史
    append_history(name, role, content)  # 追加历史 (滑动窗口)
    clear_history(name) # 清除历史
```

NPC 角色卡结构参考 `data/npcs/老马.yaml`。

### state.py — 统一状态机

玩家和每个 NPC 各持一份独立实例。

```python
class StateMachine:
    max_hp: int          # 由属性计算 (strength*2 + willpower)
    hp: int              # 当前 HP (0 = 死亡)
    alive: bool          # hp > 0

    # 情绪:  calm → wary → hostile
    # 信任:  0.0 ~ 1.0
    # 体力:  fresh → tired → exhausted

    get_state() → dict  # 返回全部维度
    apply(trigger)       # 触发规则 ("betrayed"/"helped"/"combat"/...)
    take_damage(n) → str # 扣血，返回 "alive" 或 "dead"
    restore_hp(n)        # 回血 (上限 max_hp)
```

触发规则：

| 触发词 | 效果 |
|--------|------|
| betrayed | 情绪+1, 信任-0.1 |
| helped | 情绪-1, 信任+0.1 |
| combat | 体力-1 |
| rested | 体力+1 |
| threatened | 情绪+1 |
| gifted | 信任+0.1 |

### llm.py — LLM 封装

```python
class LLM:
    chat(system, messages) → str      # 对话 (自动重试1次)
    extract_memory(dialogue) → str    # 从对话抽取关键事件
```

使用 OpenAI SDK 调用 DeepSeek API。API key 从 `DEEPSEEK_API_KEY` 环境变量读取。

### memory.py — 混合记忆

```python
class MemoryStore:
    add(content, context)       # 写入 (ChromaDB + NetworkX)
    search(query) → list       # 语义检索
    get_related(id, hops=2)    # 图遍历关联记忆
    full_retrieve(query) → list # 语义 + 图遍历合并结果
```

记忆关系类型：导致了 / 关联到 / 反驳了 / 发生在...之后

### rag.py — 知识库

```python
class KnowledgeBase:
    query(query, character) → list  # 语义搜索 + 角色权限过滤
    load_from_dir(dir)              # 加载 data/knowledge/*.md
```

Markdown 文件支持 YAML front matter 标注 `known_by` 控制角色可见范围。

### dice.py / check.py / event.py — 骰子与检定

```
dice.roll("2d6+3")  → ([3, 5], 11)

difficulty_check(dc=12)  → d20 vs DC
skill_check(value=75)    → d100 vs 技能值
opposed_check(mod1, mod2) → 双方 d20 比大小

resolve_trigger("combat", character, state, context)
  → 攻击判定 → 伤害计算 → 反击判定 → 叙事输出
```

---

## 交互模式

启动：`python trpg_agent/main.py [--debug] [-c config.yaml]`

| 命令 | 效果 |
|------|------|
| 任意文本 | 行动或对话，由三轮判定管线处理 |
| 括号内文本 | 行动声明，GM 判断检定 |
| `/dice 3d6` | 投掷骰子 |
| `查看状态` | 显示角色属性和 HP/情绪/信任/体力 |
| `/clear_npc 名称` | 清除 NPC 对话记忆 |
| `/clear_npc all` | 清除全部 NPC 记忆 |
| `exit` | 结束游戏 |
| `--debug` | 每轮输出管线诊断信息 |

---

## 测试

```bash
# 确定性逻辑 (不需要 LLM/网络)
pytest tests/test_dice.py tests/test_character.py tests/test_state.py tests/test_check.py tests/test_event.py -v

# NPC 模块 (需要 ChromaDB + embedding 模型)
pytest tests/test_npc.py -v

# 全部
pytest tests/ -v
```

确定性逻辑必测 (119 项已覆盖)。LLM 输出和 ChromaDB/embedding 相关测试需要模型文件已缓存。

---

## 关键设计决策

1. **三轮判定优于单词匹配：** 正则快速过滤 → embedding 处理同义词 → LLM 兜底。避免穷举关键词，也避免每次都调 LLM。

2. **NPC 独立 Agent：** 每个 NPC 以独立 LLM 调用扮演，system prompt 注入该 NPC 的 personality/few_shot。避免 NPC 人格被 GM prompt 稀释。

3. **统一状态机：** 玩家和 NPC 共享同一套 StateMachine 模板，各自独立实例。HP 由属性公式计算，战斗有伤害和反击闭环。

4. **ChromDB 三库分离：** memories / knowledge / npcs 使用独立 collection 和目录，互不干扰。

5. **YAGNI 优先：** 没做存档系统、没做多人、没做 UI。CLI + Rich 够用。
