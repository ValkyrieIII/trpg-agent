# TRPG Agent v2 实施计划 — 交互模型反转

> **给 Claude：** 必须使用 `superpowers:executing-plans` 子技能，按任务逐项执行本计划。

**目标：** 将交互模型从「AI 扮演艾琳与玩家对话」反转为「玩家扮演艾琳，AI 担任 GM + NPC 扮演」

**架构方案：** GM Agent 接收玩家输入，判断是叙事/检定/NPC 回应，按需调用独立 NPC Agent。玩家角色卡只保留属性/技能，NPC 角色卡增加扮演层（personality/few_shot）并由 ChromaDB 持久化。

**技术栈：** Python + OpenAI SDK (DeepSeek) + ChromaDB + NetworkX + Rich + PyYAML

---

## 任务拆解

### 任务 1：重构 Character — 移除人格层，改为玩家角色

**涉及文件：**
- 修改：`trpg_agent/character.py`
- 修改：`tests/test_character.py`

**步骤 1：先更新测试文件**

`tests/test_character.py` 中所有 fixture 的 `character` key 改为 `player`，去掉 `personality`/`few_shot` 字段。

fixture `valid_character_yaml` 变为：
```python
@pytest.fixture
def valid_character_yaml(tmp_path):
    data = {
        "player": {
            "name": "艾琳",
            "core": [
                "北方荒原的游侠",
                "曾独自在荒野生存十年",
                "沉默寡言但行动敏锐",
            ],
            "attributes": {
                "strength": 14, "agility": 18,
                "intelligence": 12, "willpower": 15,
            },
            "skills": [
                {"name": "追踪", "value": 75},
                {"name": "弓箭", "value": 80},
                {"name": "野外生存", "value": 90},
            ],
        }
    }
    path = tmp_path / "character.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return str(path)
```

fixture `minimal_character_yaml` 变为：
```python
@pytest.fixture
def minimal_character_yaml(tmp_path):
    data = {
        "player": {
            "name": "测试角色",
            "core": ["只是一个测试角色"],
            "attributes": {"strength": 10},
        }
    }
    path = tmp_path / "minimal.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return str(path)
```

**验证失败的测试也要更新：**
- `test_missing_name` — `character` → `player`，去掉 personality
- `test_missing_core` — `character` → `player`，去掉 personality
- `test_missing_personality` — 删除这个测试（personality 不再是必填字段）
- `test_missing_attributes` — `character` → `player`
- `test_missing_character_section` — 检查 key 名改为 `player`
- 删除 `TestBuildPersonalityPrompt` 整个 class（personality 相关 prompt 归 NPC 管）
- 删除 `TestBuildStatePrompt` 整个 class（状态 prompt 归 GM 管）
- 更新 `TestLoadCharacter` 中引用 `c.personality`、`c.few_shot` 的断言 — 移除或改为检查这些属性不存在
- `TestSummary` 保持不变

运行：`pytest tests/test_character.py -v`
预期：**失败（FAIL）**，因为 `character.py` 还没改

**步骤 2：运行测试确认失败**

运行：`pytest tests/test_character.py -v`
预期：类似 `缺少 'player' 字段` 的错误

**步骤 3：修改 character.py**

```python
# Character dataclass — 移除 personality 和 few_shot
@dataclass
class Character:
    name: str
    core: List[str]
    attributes: Dict[str, int] = field(default_factory=dict)
    skills: List[Dict[str, Any]] = field(default_factory=list)
```

移除 `build_personality_prompt()` 和 `build_state_prompt()` 方法。

`load_character()` 中：
- `data["character"]` → `data["player"]`
- 去掉 personality 和 few_shot 的校验和加载

`_require_field` 只检查 `name`、`core`、`attributes`（player 角色卡必填项）。

运行：`pytest tests/test_character.py -v`
预期：**全部通过（PASS）**

**步骤 4：确认测试通过**

运行：`pytest tests/test_character.py -v`
预期：全部 PASS

**步骤 5：提交变更**

```bash
git add trpg_agent/character.py tests/test_character.py
git commit -m "refactor: strip personality/few_shot from Character, make it player-only"
```

---

### 任务 2：更新 config.yaml 结构

**涉及文件：**
- 修改：`config.yaml`

**步骤 1：重写 config.yaml**

`character` → `player`，去掉 personality/few_shot/emotion_map/catchphrases，新增 `world` 段落。

```yaml
player:
  name: "艾琳"
  core:
    - "北方荒原的游侠"
    - "曾独自在荒野生存十年"
    - "沉默寡言但行动敏锐，习惯在开口前先观察"

  attributes:
    strength: 14
    agility: 18
    intelligence: 12
    willpower: 15

  skills:
    - name: "追踪"
      value: 75
    - name: "弓箭"
      value: 80
    - name: "野外生存"
      value: 90

world:
  name: "北境荒原"
  description: >
    被永冻魔法笼罩的辽阔土地。栖息着远古冰龙"霜喉"，
    每三年出现一次"蓝月潮汐"，魔法能量异常活跃，
    散落着古代瓦尔文明的遗迹。

knowledge:
  files:
    - "data/knowledge"
```

**步骤 2：提交变更**

```bash
git add config.yaml
git commit -m "refactor: restructure config.yaml to player + world"
```

---

### 任务 3：新增 npc.py — NPC 角色卡 + ChromaDB 持久化

**涉及文件：**
- 新建：`trpg_agent/npc.py`
- 新建：`tests/test_npc.py`

**步骤 1：先写测试**

`tests/test_npc.py`：

```python
"""Tests for NPC module — NPCCharacter loading and NPCStore persistence."""

import pytest
import yaml
from trpg_agent.npc import NPCCharacter, NPCStore


# ---------- NPCCharacter ----------

class TestNPCCharacter:
    def test_load_npc_from_yaml(self, tmp_path):
        data = {
            "name": "老马",
            "core": ["北境小镇酒馆老板", "年轻时是冒险者"],
            "attributes": {"strength": 13, "agility": 10},
            "skills": [{"name": "交涉", "value": 70}],
            "personality": {
                "tone": "热情粗犷",
                "verbal_tics": "每句话结尾加'是吧'",
                "emotion_map": {"anger": "拍桌子"},
                "catchphrases": ["我跟你说个秘密，是吧。"],
            },
            "few_shot": [
                {"input": "最近有什么新鲜事？", "output": "嘿，来得正好！是吧。"},
            ],
        }
        path = tmp_path / "npc.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        npc = NPCCharacter.load(path)
        assert npc.name == "老马"
        assert npc.personality["tone"] == "热情粗犷"
        assert len(npc.few_shot) == 1

    def test_build_personality_prompt(self, tmp_path):
        data = {
            "name": "老马",
            "core": ["酒馆老板"],
            "attributes": {"strength": 10},
            "skills": [],
            "personality": {
                "tone": "热情", "verbal_tics": "是吧",
                "emotion_map": {}, "catchphrases": ["嘿！"],
            },
            "few_shot": [],
        }
        path = tmp_path / "npc2.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        npc = NPCCharacter.load(path)
        prompt = npc.build_personality_prompt()
        assert "酒馆老板" in prompt
        assert "热情" in prompt
        assert "嘿！" in prompt

    def test_build_state_prompt(self, tmp_path):
        data = {
            "name": "老马",
            "core": ["酒馆老板"],
            "attributes": {"strength": 10},
            "skills": [],
            "personality": {
                "tone": "热情", "verbal_tics": "是吧",
                "emotion_map": {"anger": "拍桌子"},
                "catchphrases": [],
            },
            "few_shot": [],
        }
        path = tmp_path / "npc3.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        npc = NPCCharacter.load(path)
        prompt = npc.build_state_prompt({"emotion": "anger", "trust": 0.3, "stamina": "fresh"})
        assert "拍桌子" in prompt


# ---------- NPCStore ----------

class TestNPCStore:
    def test_save_and_find_by_name(self):
        store = NPCStore()
        npc = NPCCharacter(
            name="测试NPC",
            core=["测试"],
            attributes={"str": 10},
            personality={"tone": "平淡", "verbal_tics": "", "emotion_map": {}, "catchphrases": []},
        )
        store.save(npc)
        found = store.find_by_name("测试NPC")
        assert found is not None
        assert found.name == "测试NPC"

    def test_search_by_query(self):
        store = NPCStore()
        npc = NPCCharacter(
            name="铁匠老王",
            core=["镇上的铁匠", "打造武器四十年"],
            attributes={"str": 16},
            personality={"tone": "粗鲁", "verbal_tics": "", "emotion_map": {}, "catchphrases": []},
        )
        store.save(npc)
        results = store.search("武器铁匠")
        assert len(results) > 0

    def test_all_npcs(self):
        store = NPCStore()
        npc1 = NPCCharacter(
            name="A", core=["..."],
            attributes={"str": 10},
            personality={"tone": "", "verbal_tics": "", "emotion_map": {}, "catchphrases": []},
        )
        npc2 = NPCCharacter(
            name="B", core=["..."],
            attributes={"str": 10},
            personality={"tone": "", "verbal_tics": "", "emotion_map": {}, "catchphrases": []},
        )
        store.save(npc1)
        store.save(npc2)
        names = [n.name for n in store.all()]
        assert "A" in names
        assert "B" in names

    def test_create_dynamic_npc(self):
        store = NPCStore()
        npc = store.create(
            name="流浪商人",
            core=["游走于北境各城镇的商人"],
            attributes={"strength": 8, "agility": 12},
            skills=[{"name": "交涉", "value": 80}],
            personality={"tone": "油滑", "verbal_tics": "", "emotion_map": {}, "catchphrases": []},
        )
        assert npc.name == "流浪商人"
        found = store.find_by_name("流浪商人")
        assert found is not None
```

**步骤 2：运行测试确认失败**

```bash
pytest tests/test_npc.py -v
```
预期：**FAIL**，模块不存在

**步骤 3：实现 npc.py**

```python
"""NPC module — NPCCharacter with personality + NPCStore for persistence."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
import yaml
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from trpg_agent.character import Character


@dataclass
class NPCCharacter(Character):
    """NPC character card — extends Character with personality layer."""
    personality: Dict[str, Any] = field(default_factory=dict)
    few_shot: List[Dict[str, str]] = field(default_factory=list)

    def build_personality_prompt(self) -> str:
        """Assemble NPC personality system prompt."""
        parts: List[str] = []
        parts.append("【角色背景】")
        parts.extend(self.core)
        parts.append("")
        parts.append("【说话方式】")
        parts.append(f"语调：{self.personality.get('tone', '正常')}")
        parts.append(f"语言习惯：{self.personality.get('verbal_tics', '无')}")

        catchphrases = self.personality.get("catchphrases", [])
        if catchphrases:
            parts.append("")
            parts.append("【口头禅】")
            for cp in catchphrases:
                parts.append(f"- {cp}")

        if self.few_shot:
            parts.append("")
            parts.append("【对话示例】")
            for example in self.few_shot:
                parts.append(f"玩家：{example['input']}")
                parts.append(f"你：{example['output']}")
                parts.append("")

        return "\n".join(parts)

    def build_state_prompt(self, state: dict) -> str:
        """Build state block from NPC's emotion_map."""
        emotion = state.get("emotion", "calm")
        trust = state.get("trust", 0.5)
        stamina = state.get("stamina", "fresh")
        emotion_map = self.personality.get("emotion_map", {})
        behaviour = emotion_map.get(emotion, "正常反应")
        lines = [
            "【当前状态】",
            f"情绪：{emotion}",
            f"信任度：{trust}",
            f"体力：{stamina}",
            f"行为表现：{behaviour}",
        ]
        return "\n".join(lines)

    @classmethod
    def load(cls, path: str) -> NPCCharacter:
        """Load NPC from a YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"错误：找不到 NPC 配置文件 {path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"错误：NPC 配置文件格式有误 — {e}")
            sys.exit(1)

        _require = _require_field_npc  # ... validation

        _require(data, "name", "NPC名称")
        _require(data, "core", "NPC背景")
        _require(data, "personality", "NPC人格设置")

        return cls(
            name=data["name"],
            core=data["core"],
            attributes=data.get("attributes", {}),
            skills=data.get("skills", []),
            personality=data["personality"],
            few_shot=data.get("few_shot", []),
        )


def _require_field_npc(data: dict, key: str, cn_name: str) -> None:
    if key not in data or not data[key]:
        print(f"错误：NPC 配置缺少必填字段「{cn_name}」({key})")
        sys.exit(1)


class NPCStore:
    """NPC persistence layer — ChromaDB for semantic search + per-NPC history.

    Parameters
    ----------
    persist_dir : str
        ChromaDB 持久化目录（默认 ``"data/chroma/npcs"``）。
    """

    def __init__(self, persist_dir: str = "data/chroma/npcs") -> None:
        self._persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name="npcs",
            embedding_function=self._embedding_fn,
        )
        self._npcs: Dict[str, NPCCharacter] = {}
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self._max_history = 10
        self._load_all()

    def save(self, npc: NPCCharacter) -> None:
        """Persist an NPC to ChromaDB (upsert by name)."""
        doc = f"{npc.name}: " + " ".join(npc.core)
        meta = {
            "name": npc.name,
            "attributes": json.dumps(npc.attributes, ensure_ascii=False),
            "skills": json.dumps(npc.skills, ensure_ascii=False),
            "personality": json.dumps(npc.personality, ensure_ascii=False),
            "few_shot": json.dumps(npc.few_shot, ensure_ascii=False),
        }
        # Remove old entry if exists, then add
        existing = self._collection.get(where={"name": npc.name})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
        from uuid import uuid4
        self._collection.add(
            ids=[uuid4().hex[:8]],
            documents=[doc],
            metadatas=[meta],
        )
        self._npcs[npc.name] = npc

    def find_by_name(self, name: str) -> Optional[NPCCharacter]:
        """Look up NPC by exact name (in-memory cache)."""
        return self._npcs.get(name)

    def search(self, query: str, n: int = 5) -> List[NPCCharacter]:
        """Semantic search for NPCs."""
        raw = self._collection.query(query_texts=[query], n_results=n)
        results = []
        ids_list = raw.get("ids", [[]])
        if not ids_list or not ids_list[0]:
            return results
        for i in range(len(ids_list[0])):
            meta = raw["metadatas"][0][i]
            name = meta["name"]
            if name in self._npcs:
                results.append(self._npcs[name])
        return results

    def all(self) -> List[NPCCharacter]:
        """Return all known NPCs."""
        return list(self._npcs.values())

    def create(self, name: str, core: List[str], attributes: Dict[str, int],
               skills: List[Dict] = None, personality: Dict = None) -> NPCCharacter:
        """Dynamically create and persist a new NPC."""
        npc = NPCCharacter(
            name=name, core=core, attributes=attributes,
            skills=skills or [],
            personality=personality or {"tone": "", "verbal_tics": "", "emotion_map": {}, "catchphrases": []},
        )
        self.save(npc)
        return npc

    def get_history(self, name: str) -> List[Dict[str, str]]:
        """Get per-NPC conversation history (sliding window)."""
        if name not in self._histories:
            self._histories[name] = []
        return self._histories[name]

    def append_history(self, name: str, role: str, content: str) -> None:
        """Append a turn to NPC conversation history."""
        if name not in self._histories:
            self._histories[name] = []
        self._histories[name].append({"role": role, "content": content})
        if len(self._histories[name]) > self._max_history * 2:
            self._histories[name] = self._histories[name][-self._max_history * 2:]

    def _load_all(self) -> None:
        """Load all NPCs from ChromaDB into memory cache."""
        try:
            raw = self._collection.get()
        except Exception:
            return
        if not raw["ids"]:
            return
        for i in range(len(raw["ids"])):
            meta = raw["metadatas"][i]
            npc = NPCCharacter(
                name=meta["name"],
                core=[raw["documents"][i]],
                attributes=json.loads(meta.get("attributes", "{}")),
                skills=json.loads(meta.get("skills", "[]")),
                personality=json.loads(meta.get("personality", "{}")),
                few_shot=json.loads(meta.get("few_shot", "[]")),
            )
            self._npcs[npc.name] = npc
```

**步骤 4：运行测试确认通过**

```bash
pytest tests/test_npc.py -v
```
预期：全部 PASS

**步骤 5：提交变更**

```bash
git add trpg_agent/npc.py tests/test_npc.py
git commit -m "feat: add NPC module with ChromaDB persistence"
```

---

### 任务 4：创建初始 NPC 角色卡

**涉及文件：**
- 新建：`data/npcs/老马.yaml`
- 新建：`data/npcs/.gitkeep`（可选）

**步骤 1：写 NPC YAML**

`data/npcs/老马.yaml`：
```yaml
name: "老马"
core:
  - "北境小镇'霜降'的酒馆老板"
  - "年轻时曾是冒险者，膝盖中过一箭后退隐"
  - "消息灵通，来往旅客的消息都逃不过他的耳朵"

attributes:
  strength: 13
  agility: 10
  intelligence: 14
  willpower: 12

skills:
  - name: "交涉"
    value: 70
  - name: "酿酒"
    value: 85

personality:
  tone: "热情粗犷，爱讲故事，偶尔吹牛"
  verbal_tics: "每句话末尾爱加'是吧'"
  emotion_map:
    anger: "拍桌子，嗓门变大"
    fear: "压低声音，四处张望"
    trust: "凑近压低声音分享秘密，是吧。"
  catchphrases:
    - "我跟你说个秘密，是吧。"
    - "当年老子冒险的时候..."

few_shot:
  - input: "最近有什么新鲜事？"
    output: "嘿，你来得正好！昨晚上矿洞那边又传出怪声了，是吧。"
  - input: "你这里有什么好酒？"
    output: "北境最好的麦酒！喝了我的酒，冰龙都不怕，是吧。"
  - input: "谢谢你，老马。"
    output: "客气啥！下次带点外面的故事来换酒喝，是吧。"
```

**步骤 2：提交变更**

```bash
git add data/npcs/
git commit -m "feat: add sample NPC 老马 (innkeeper)"
```

---

### 任务 5：重构 GameMaster — GM prompt + 调度逻辑

**涉及文件：**
- 修改：`trpg_agent/game_master.py`

这是核心改动，不需要写新的 pytest（LLM 输出手动验证），但要确保已有测试不坏。

**步骤 1：重写 GM system prompt 和相关方法**

关键改动点：

1. **`__init__` 中**：
   - `self.character` 改为 `self.player`（名字也改了）
   - 新增 `self.npc_store = NPCStore()`
   - 新增 `self.scene_npcs: List[str] = []`（当前场景 NPC 名列表）
   - 新增 `self.player_history: List[Dict[str, str]] = []`（玩家-GM 对话历史）

2. **新 GM system prompt**（替换旧的 `_GM_SYSTEM` + `_call_gm`）：
```python
_GM_SYSTEM = (
    "你是 TRPG 地下城主(Game Master)。\n"
    "玩家扮演 {player_name}，{player_desc}\n\n"
    "## 职责\n"
    "1. 叙述场景 — 用第三人称客观描述玩家看到、听到、感受到的一切。\n"
    "2. 判断检定 — 玩家用括号声明行动时（如'(我拔出短刀)'），判断是否需要检定并执行。纯扮演动作（微笑、点头）无需检定。\n"
    "3. 扮演 NPC — 当玩家对 NPC 说话或互动时，你需要以该 NPC 的身份回应。\n\n"
    "## 规则\n"
    "- 不要替玩家角色说话或替 ta 做决定。\n"
    "- 不要描述玩家角色的内心感受（'你觉得...'），只描述客观事实。\n"
    "- 每段叙述不超过 150 字。\n"
    "- 当玩家询问不在场的 NPC 时，直接说明 ta 不在。\n"
    "- 你可以在叙事中引入新 NPC，引入后该 NPC 将持续存在于世界中。\n\n"
    "## 当前场景已知 NPC\n"
    "{scene_npcs}\n\n"
    "## 玩家角色卡\n"
    "{player_card}"
)
```

3. **新 GM 调用方法**（替换 `_call_gm` + `_handle_dialogue` 的 GM 部分）：
```python
def _call_gm(self, user_input: str, knowledge: list[str], memories: list[dict]) -> dict:
    """GM Agent: 分析输入，返回 {narration, check_result, responding_npc}."""
    # 构建 prompt
    player_desc = "，".join(self.player.core)
    scene_npcs = "\n".join(f"- {n}" for n in self.scene_npcs) if self.scene_npcs else "（暂无已知 NPC）"
    player_card = self.player.summary()

    system = self._GM_SYSTEM.format(
        player_name=self.player.name,
        player_desc=player_desc,
        scene_npcs=scene_npcs,
        player_card=player_card,
    )

    # 检索结果注入
    if knowledge:
        system += f"\n\n【相关知识】\n" + "\n".join(knowledge[:3])
    if memories:
        system += f"\n\n【相关记忆】\n" + "\n".join(m["content"] for m in memories[:3])

    # 检定规则注入
    system += "\n\n## 可用检定\n"
    for pattern, check_type, skill_or_dc, mod in _ACTION_RULES:
        system += f"- {pattern}: {check_type}({skill_or_dc})\n"

    system += (
        "\n## 输出格式\n"
        "返回 JSON 格式（不要加 markdown 代码块标记）：\n"
        "{\n"
        '  "narration": "场景叙述文本（可为空字符串）",\n'
        '  "check": null 或 {"type": "skill"/"check", "target": 技能名或DC值, "mod": 修正值, "desc": "行动描述"},\n'
        '  "responding_npc": "NPC名称 或 null",\n'
        '  "new_npc": null 或 {"name": "...", "core": ["..."], "attributes": {...}, "personality": {...}}\n'
        "}"
    )

    messages = list(self.player_history[-10:])
    messages.append({"role": "user", "content": user_input})

    if self._llm_available and self.llm:
        try:
            raw = self.llm.chat(system=system, messages=messages)
            return self._parse_gm_response(raw)
        except Exception:
            pass
    return {"narration": "", "check": None, "responding_npc": None, "new_npc": None}
```

4. **JSON 解析 + 降级**：
```python
def _parse_gm_response(self, raw: str) -> dict:
    """Parse GM JSON response, fallback to regex if JSON fails."""
    import json as json_module
    try:
        # Strip potential markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
        result = json_module.loads(cleaned)
        return {
            "narration": result.get("narration", ""),
            "check": result.get("check"),
            "responding_npc": result.get("responding_npc"),
            "new_npc": result.get("new_npc"),
        }
    except (json_module.JSONDecodeError, ValueError):
        # Fallback: treat entire response as narration
        return {"narration": raw, "check": None, "responding_npc": None, "new_npc": None}
```

5. **重写 `_handle_dialogue`**：
```python
def _handle_dialogue(self, user_input: str) -> str:
    name = self.player.name

    # ---- Step 0: 硬编码行动匹配 ----
    action = self._match_action(user_input)

    # ---- Step 1: 检索 ----
    memories = self.memory.full_retrieve(user_input)
    knowledge = self.knowledge.query(user_input, name)

    # ---- Step 2: GM Agent 分析 ----
    gm_result = self._call_gm(user_input, knowledge, memories)

    # ---- Step 3: 执行检定（GM 判断需要 or 硬编码匹配到） ----
    check_result = None
    if gm_result.get("check"):
        check_data = gm_result["check"]
        if check_data["type"] == "skill":
            check_result = skill_check(check_data["target"], check_data.get("mod", 0))
        else:
            check_result = difficulty_check(check_data["target"], check_data.get("mod", 0))
    elif action:
        check_result = self._execute_check(action)

    # ---- Step 4: 构建 GM 叙事部分 ----
    parts = []
    gm_narration = gm_result.get("narration", "")
    if gm_narration:
        parts.append(f"[GM] {gm_narration}")
    if check_result:
        parts.append(check_result["narrative"])

    # ---- Step 5: 处理新 NPC 创建 ----
    new_npc = gm_result.get("new_npc")
    if new_npc:
        npc = self.npc_store.create(
            name=new_npc["name"],
            core=new_npc.get("core", []),
            attributes=new_npc.get("attributes", {}),
            personality=new_npc.get("personality", {}),
        )
        npc_name = npc.name
        if npc_name not in self.scene_npcs:
            self.scene_npcs.append(npc_name)
    else:
        npc_name = gm_result.get("responding_npc")

    # ---- Step 6: NPC Agent 调用 ----
    npc_reply = ""
    if npc_name:
        npc = self.npc_store.find_by_name(npc_name)
        if npc is None:
            # 尝试语义搜索
            results = self.npc_store.search(npc_name)
            if results:
                npc = results[0]

        if npc:
            # 确保 NPC 在场
            if npc_name not in self.scene_npcs:
                self.scene_npcs.append(npc_name)

            npc_system = npc.build_personality_prompt()
            npc_system += "\n\n" + npc.build_state_prompt(
                {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}
            )

            npc_messages = list(self.npc_store.get_history(npc_name))
            npc_messages.append({
                "role": "user",
                "content": f"{self.player.name}对你说：{user_input}",
            })

            if self._llm_available and self.llm:
                try:
                    npc_reply = self.llm.chat(system=npc_system, messages=npc_messages)
                except Exception:
                    npc_reply = "..."

            if npc_reply:
                self.npc_store.append_history(npc_name, "user",
                    f"{self.player.name}: {user_input}")
                self.npc_store.append_history(npc_name, "assistant", npc_reply)
                parts.append(f"[{npc_name}] {npc_reply}")

    response = "\n\n".join(parts) if parts else f"[GM] （沉默）"

    # ---- Step 7: 状态更新 ----
    self._update_state_from_keywords(f"{user_input} {gm_narration} {npc_reply}")
    if action and check_result and check_result.get("stamina_cost"):
        self.state.apply("combat")

    # ---- Step 8: 记忆记录 ----
    if self._llm_available and self.llm:
        try:
            extracted = self.llm.extract_memory(
                f"玩家（{name}）：{user_input}\n{npc_name or 'GM'}：{npc_reply or gm_narration}"
            )
        except Exception:
            extracted = ""
    else:
        extracted = ""
    if extracted:
        self.memory.add(
            content=extracted,
            context={"emotion": self.state.get_state()["emotion"]},
        )

    # ---- Step 9: 更新玩家-GM 对话历史 ----
    self.player_history.append({"role": "user", "content": user_input})
    self.player_history.append({"role": "assistant", "content": response})
    if len(self.player_history) > self.max_history * 2:
        self.player_history = self.player_history[-self.max_history * 2:]

    return response
```

6. **删除/注释掉旧方法**：`_call_gm` 旧版本、`_GM_SYSTEM` 旧版本、`_handle_dialogue` 旧版本中角色 agent 相关逻辑

7. **更新 `process()` 中调用 `gm.character.name`** → `gm.player.name`

**步骤 2：运行已有测试确保不破坏**

```bash
pytest tests/ -v
```
预期：character/npc/dice/state/memory/rag/check/event 全部 PASS

**步骤 3：提交变更**

```bash
git add trpg_agent/game_master.py
git commit -m "feat: rewrite GM to player-driven model with NPC dispatch"
```

---

### 任务 6：更新 main.py — 交互文案

**涉及文件：**
- 修改：`trpg_agent/main.py`

**步骤 1：修改欢迎文案**

- `gm.character.name` → `gm.player.name`
- 欢迎面板文字改为玩家视角：

```python
character_name = gm.player.name

console.print(
    Panel(
        f"[bold cyan]{character_name}[/bold cyan]，你站在北境荒原的边缘...\n\n"
        "可用命令：\n"
        "  [green]/dice <表达式>[/green]  投掷骰子（如 [green]/dice 3d6[/green]）\n"
        "  [green]exit[/green] / [green]quit[/green] / [green]退出[/green]  结束游戏\n\n"
        "输入你想做的事。用括号声明行动，如 [dim](我拔出弓箭)[/dim]。\n"
        "也可以直接对 NPC 说话。",
        title="[bold]TRPG Agent — 北境荒原[/bold]",
        subtitle=f"你扮演 {character_name}",
        border_style="bright_blue",
    )
)
```

- 回复面板 title 改为 `[bold]GM[/bold]`（或 `[bold]北境荒原[/bold]`），不再显示角色名

**步骤 2：提交变更**

```bash
git add trpg_agent/main.py
git commit -m "refactor: update main.py for player-driven interaction"
```

---

### 任务 7：端到端 smoke test

**涉及文件：** 无（手动验证）

**步骤 1：启动应用**

```bash
python trpg_agent/main.py
```

**步骤 2：测试以下场景**

1. 纯对话："老马，给我来杯酒"
   - 预期：GM 叙事酒馆场景 + 老马以 NPC 身份回应
2. 行动检定："(我拔出短刀，潜行靠近矿洞入口)"
   - 预期：GM 判断检定，执行检定，叙述结果
3. 无 NPC 的行动："(我环顾四周，观察酒馆里的客人)"
   - 预期：GM 直接叙事描述，无 NPC 回应
4. 骰子命令：`/dice 3d6`
   - 预期：正常显示骰子结果
5. 角色信息：`查看状态`
   - 预期：显示玩家角色属性和状态

**步骤 3：确认无崩溃，NPC 语气一致**

---

### 任务 8：更新 GAMEPLAY.md

**涉及文件：**
- 修改：`docs/GAMEPLAY.md`

**步骤 1：更新文档**

- 把"AIR 琳"的描述改为"你扮演艾琳"
- 更新交互说明为玩家视角
- 添加 NPC 互动说明

**步骤 2：提交变更**

```bash
git add docs/GAMEPLAY.md
git commit -m "docs: update gameplay guide for v2 player-driven model"
```

---

## 验证方式

全部完成后运行完整测试套件：

```bash
pytest tests/ -v
```
预期：所有确定性逻辑测试 PASS。

端到端手动 smoke test 覆盖对话 + 检定 + NPC 回复 + 骰子。

## 风险与注意事项

- **GM JSON 输出不可靠：** DeepSeek V4 Flash 可能不严格遵守 JSON 格式，已通过 `_parse_gm_response` 降级为全文当作 narration 处理
- **已有测试破坏：** `test_character.py` 修改较大，其余测试（dice/state/memory/rag/check/event）不应受影响
- **NPC per-agent history 不持久化：** 会话内维护，重启后清空。后续可考虑持久化
- **player_history vs history 字段更名：** `self.history` 不再使用，改为 `self.player_history`
