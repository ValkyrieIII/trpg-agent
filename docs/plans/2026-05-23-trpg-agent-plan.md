# TRPG Agent 实施计划

> **给 Claude：** 必须使用 `superpowers:executing-plans` 子技能，按任务逐项执行本计划。代码由执行 agent 自行生成，本计划只定义模块职责、接口约定和逻辑约束。

**目标：** 一天内搭建一个带长时记忆和人格化角色的跑团 CLI agent。

**架构方案：** Anthropic SDK + ChromaDB + NetworkX + Rich CLI，分层架构，骰子/角色/状态机/GM/事件 各层独立，确定性逻辑 TDD，LLM 交互手动验证。

**技术栈：** Python 3.10+, anthropic, chromadb, networkx, sentence-transformers, rich, pyyaml, pytest

---

## 任务拆解

### 任务 1：项目初始化

**涉及文件：**
- 新建：`requirements.txt`
- 新建：`config.yaml`
- 新建：`.gitignore`
- 新建：`data/knowledge/.gitkeep`
- 新建：`trpg_agent/__init__.py`

**步骤 1：创建 requirements.txt**

内容：anthropic, chromadb, networkx, sentence-transformers, rich, pyyaml, pytest。主版本号锁定（避免 breaking changes）。

**步骤 2：创建 .gitignore**

忽略：`data/chroma/`（ChromaDB 持久化数据）、`__pycache__/`、`.pytest_cache/`、模型缓存目录。

**步骤 3：创建项目目录结构**

```
trpg_agent/
tests/
data/knowledge/
```

**步骤 4：创建 trpg_agent/__init__.py**（空模块标记）

**步骤 5：创建 config.yaml**

内容参照设计文档中角色卡结构，包含 character 完整配置 + knowledge.files 列表（指定要加载的知识文件路径）。

**步骤 6：安装依赖并初始化 git**

预期：全部安装成功，无报错。

---

### 任务 2：骰子系统 dice.py

**涉及文件：**
- 新建：`trpg_agent/dice.py`
- 新建：`tests/test_dice.py`

**模块职责：** 骰子表达式解析和投掷，作为所有随机数生成的统一入口。项目中所有需要随机数的模块（检定、事件等）都应调用 `dice.roll()`，不直接使用 `random`。

**接口：**
- `parse_dice(expr: str) -> tuple[int, int, int]` — 解析 "ndm+k" 格式，返回 (数量, 面数, 修正值)
- `roll(expr: str) -> tuple[list[int], int]` — 投掷并返回 (各骰结果列表, 含修正的总值)

**支持的表达式：** `d20`、`d100`、`3d6`、`2d6+3`。注意兼容用户输入中的空格（如 `"2d6 + 3"`）。

**测试覆盖：**
- 各表达式解析正确性
- 投掷结果范围验证（每个骰子结果在 [1, sides] 区间）
- 无效表达式抛出异常
- 带空格表达式能正常解析

---

### 任务 3：角色系统 character.py

**涉及文件：**
- 新建：`trpg_agent/character.py`
- 新建：`tests/test_character.py`

**模块职责：** 角色卡数据模型，从 YAML 加载并提供 prompt 拼接方法。

**数据模型：** Character 包含 name, core（角色背景描述列表）, personality（tone/verbal_tics/emotion_map/catchphrases）, few_shot（对话示例列表）, attributes（属性字典）, skills（技能字典）。

**接口：**
- `load_character(config_path: str) -> Character` — 从 YAML 加载，启动时校验必填字段（name, core, personality, attributes），缺失时给出友好错误提示而非 Python traceback
- `Character.build_personality_prompt() -> str` — 拼接人格 prompt：core + tone + catchphrases + few_shot 示例格式化
- `Character.build_state_prompt(state: dict) -> str` — 根据当前状态（情绪/信任/体力）从 emotion_map 获取对应行为描述并格式化
- `Character.summary() -> str` — 返回角色属性/技能摘要文本
- `Character.load(config_path) -> Character` — 类方法，等价于 load_character

**逻辑约束：**
- `build_personality_prompt` 必须包含 few_shot 示例（格式化为对话示例注入 prompt）
- 校验时对每个必填字段逐一检查，给出明确缺失提示

**测试覆盖：**
- 从 YAML 加载完整角色
- 缺字段时给出友好错误
- prompt 拼接包含全部必要元素
- 状态 prompt 正确映射 emotion → 行为描述

---

### 任务 4：状态机 state.py

**涉及文件：**
- 新建：`trpg_agent/state.py`
- 新建：`tests/test_state.py`

**模块职责：** 轻量规则驱动状态机，追踪角色的情绪/信任/体力三维度。不使用 LLM，纯规则触发。

**维度定义：**
- 情绪：calm → wary → hostile（三级有序）
- 信任：0.0 ~ 1.0（浮点连续）
- 体力：fresh → tired → exhausted（三级有序）

**触发规则：**
```
betrayed:  emotion+1, trust-0.1
helped:    emotion-1, trust+0.1
combat:    体力消耗一级
rested:    体力恢复一级
threatened: emotion+1
gifted:    trust+0.1
```

**接口：**
- `StateMachine.get_state() -> dict` — 返回当前状态 {"emotion": str, "trust": float, "stamina": str}
- `StateMachine.apply(trigger: str)` — 应用触发词，更新内部状态

**逻辑约束：**
- 所有维度到达上下界后不再越界（emotion 已 hostile 再 +1 仍为 hostile；stamina 已 fresh 再 restore 仍为 fresh；trust 已达 1.0 再 +0.1 仍为 1.0）
- 未识别的 trigger 静默忽略，不报错

**测试覆盖：**
- 初始状态值
- 情绪上升路径（calm → wary → hostile）
- 情绪下降路径（hostile → wary → calm）
- 信任增减及边界（0.0 不越界，1.0 不越界）
- 体力消耗与恢复及边界
- 已到边界后再触发不越界
- 未知 trigger 无变化

---

### 任务 5：LLM 封装 llm.py

**涉及文件：**
- 新建：`trpg_agent/llm.py`

**模块职责：** Claude API 封装，提供对话和事件抽取两个能力。

**接口：**
- `LLM(model: str)` — 初始化，从环境变量 ANTHROPIC_API_KEY 读取 key
- `LLM.chat(system: str, messages: list[dict]) -> str` — 发送对话请求，返回回复文本
- `LLM.extract_memory(dialogue: str) -> str` — 从对话文本中抽取 1-2 条关键事件，用中文一句话概括

**逻辑约束：**
- `chat()` 必须包含重试逻辑：API 调用失败时重试一次（共两次机会），两次都失败再抛出 RuntimeError
- `extract_memory()` 需验证响应非空，空响应返回空字符串而非崩溃
- 模型默认使用 claude-sonnet-4-6

**测试：** 不写 pytest（依赖 API key），手动 smoke 验证。

---

### 任务 6：记忆系统 memory.py

**涉及文件：**
- 新建：`trpg_agent/memory.py`
- 新建：`tests/test_memory.py`

**模块职责：** 混合存储系统：ChromaDB 做语义检索入口，NetworkX 表达记忆条目间的逻辑关系。记忆条目包含时间戳、重要性、上下文等元数据。

**存储架构：**
- ChromaDB 持久化目录：`data/chroma/memories/`（与知识库的 `data/chroma/knowledge/` 分离）
- NetworkX 图：内存中维护，每次添加/链接时同步持久化到 JSON 文件，启动时加载

**记忆条目结构：** id, content, type(event/fact/emotion_peak), timestamp, importance(0-1), tags, context:{location, npcs, emotion}

**关系类型：** "导致了"、"关联到"、"反驳了"、"发生在...之后"

**接口：**
- `MemoryStore.add(content, context, importance=0.5) -> str` — 添加记忆，返回记忆 ID
- `MemoryStore.search(query, n=3) -> list[dict]` — 语义检索，返回的记忆条目需包含 id, content, importance, context
- `MemoryStore.link(id1, id2, relation)` — 建立图关系
- `MemoryStore.get_related(mem_id, hops=2) -> list[dict]` — 图遍历，返回 1-hop 至 N-hop 关联记忆
- `MemoryStore.full_retrieve(query) -> list[dict]` — 联合检索：语义检索 → 图遍历拉关联 → 合并去重 → 按 importance 降序排序返回

**逻辑约束：**
- `full_retrieve` 必须按 importance 降序排序，高重要性记忆优先注入 prompt
- 记忆条目存储时带上 importance 字段；图遍历拉回的关联记忆 importance 在原值基础上 +0.1（关联记忆更相关）
- 图和 ChromaDB 保持同步：添加记忆时同时写入两边，删除暂不支持

**测试覆盖：**
- 添加 → 语义检索来回一致性（确保写入能搜到）
- 图链接 → get_related 返回正确关联记忆
- full_retrieve 返回结果按 importance 降序
- 图遍历 hops 参数生效（1-hop vs 2-hop）

**测试注意事项：**
- 测试使用临时目录（tmp_path），避免测试数据污染和相互干扰
- 语义检索的断言使用宽松匹配（检查内容包含关键词而非精确相等），避免 embedding 模型差异导致 flaky

---

### 任务 7：RAG 知识库 rag.py

**涉及文件：**
- 新建：`trpg_agent/rag.py`
- 新建：`tests/test_rag.py`
- 新建：`data/knowledge/world.md`
- 新建：`data/knowledge/npc_innkeeper.md`

**模块职责：** 世界观知识存储和角色身份过滤检索。知识源为 `data/knowledge/` 下的 Markdown 文件，通过 YAML front matter 标注权限。

**存储架构：**
- ChromaDB 持久化目录：`data/chroma/knowledge/`（与记忆系统的 `data/chroma/memories/` 分离）
- 知识条目 metadata 包含 `known_by`（逗号分隔的角色列表）和 `category`（分类标签）

**知识文件格式（YAML front matter + Markdown）：**

每条知识使用 YAML front matter 标记：
```yaml
---
known_by: [酒馆老板, 老酒客]
category: location
---
酒馆地下有一条密道通往郊外，入口在酒窖第三个木桶后面。
```

无 front matter 的段落默认 `known_by: 所有人`。

**接口：**
- `KnowledgeBase.add_knowledge(content, known_by, category)` — 添加单条知识，`known_by` 可以是 str 或 list[str]，"所有人" 表示公开知识
- `KnowledgeBase.query(query, character, n=3, threshold=0.5) -> list[str]` — 语义检索并按角色身份过滤
- `KnowledgeBase.load_from_dir(dir_path)` — 批量加载目录下所有 .md 文件，解析 YAML front matter

**逻辑约束：**
- `query()` 的权限过滤：`character in known_by` 或 `"所有人" in known_by` 才返回
- 阈值过滤：ChromaDB 返回的相似度低于阈值的条目不注入（避免角色"知道"无关信息）。注意：ChromaDB 使用 cosine distance（越小越相似），距离值 ≤ threshold 的保留
- `load_from_dir()` 正确解析 YAML front matter，未标注 known_by 的段落默认为"所有人"
- `load_from_dir()` 在 GameMaster 初始化时必须被调用（通过 config.yaml 中的 knowledge.files 列表指定目录）

**测试覆盖：**
- 权限过滤：角色 A 能看到自己的知识，看不到仅限角色 B 的知识
- "所有人" 知识对所有角色可见
- 阈值过滤生效
- load_from_dir 正确解析 front matter 中的 known_by

**测试注意事项：**
- 测试使用临时目录（tmp_path），避免生产数据污染

---

### 任务 8：检定系统 check.py

**涉及文件：**
- 新建：`trpg_agent/check.py`
- 新建：`tests/test_check.py`

**模块职责：** 三种检定类型（技能检定/难度检定/对抗检定），复用 dice.py 的 roll() 统一生成随机数。

**接口：**
- `skill_check(skill_value, modifier=0) -> dict` — d100 检定：`d100 roll ≤ skill_value + modifier` 为成功。注意 modifier 加到 skill_value 上（正 modifier 使成功更容易），而非加到 roll 上
- `difficulty_check(dc, modifier=0) -> dict` — d20 难度检定：`d20 + modifier ≥ DC` 为成功
- `opposed_check(player_mod, opponent_mod) -> dict` — 对抗检定：双方各 d20 + 修正，比大小

**返回值格式：**
```python
{"success": bool, "detail": str, ...}  # 各类型包含各自的 roll/target 等字段
```

**逻辑约束：**
- 骰子随机数统一通过 `dice.roll()` 生成，不在 check.py 中直接调用 `random`
- 对抗检定需处理平局：当双方总分相等时，`success` 为 False，但额外返回 `tie: True` 字段。调用方可据此决定平局如何处理（重掷或以防守方胜出）
- skill_check 的 modifier 语义：正 modifier 是增益（加到目标值）

**测试覆盖：**
- 极值情况（skill_value=100 必成功，skill_value=0 必失败）
- DC=0 必成功，DC=30 必失败（d20+mod 最高约 25）
- 对抗检定平局返回 tie=True
- modifier 方向验证（正 modifier 增加成功率）

---

### 任务 9：事件系统 event.py

**涉及文件：**
- 新建：`trpg_agent/event.py`

**模块职责：** 非对话的事件判定逻辑。当玩家触发陷阱、环境交互、NPC 反应时，GM 通过此模块判定结果。本质是检定系统 + 状态机 + GM 判断的聚合层，不新增底层逻辑。

**接口：**
- `resolve_trigger(trigger_type: str, character, state: dict, context: dict) -> dict` — 事件判定入口
  - trigger_type: "trap" / "environment" / "npc_reaction" / "discovery"
  - 返回: {"outcome": str, "state_changes": list[str], "narrative": str}

**判定逻辑（确定性，不调 LLM）：**
- 陷阱触发：difficulty_check(DC=15) → 成功则"察觉并避开"，失败则 state.apply("combat") + 返回受伤叙事
- 环境交互：difficulty_check(DC 随环境难度而定) → 成功/失败叙事
- NPC 反应：根据当前 state["trust"] 阈值决定友好/中立/敌对反应
- 发现：skill_check(角色对应技能值) → 成功则返回发现叙事

**与 GM 的关系：** GM 的 `_handle_rule` 中不再使用空的 `combat_check` 占位，而是将战斗/事件类输入路由到 `event.resolve_trigger()`。

---

### 任务 10：GM 核心调度 game_master.py

**涉及文件：**
- 新建：`trpg_agent/game_master.py`

**模块职责：** 对话调度中枢。聚合所有子系统，负责意图识别、上下文组装、prompt 构建、状态更新、记忆写入、历史管理。

**意图识别（正则匹配，不调 LLM）：**

| 模式 | 意图 | 处理方式 |
|------|------|---------|
| 掷骰/骰子/d20/d100 等 | dice | 调用 dice.roll() 返回结果 |
| 查看/属性/状态/角色卡 | info | 调用 character.summary() |
| 战斗/攻击/触发/环境等 | event | 调用 event.resolve_trigger() |
| 其余 | dialogue | 走完整对话管线 |

注意：之前的 combat_check 意图替换为 event，由事件系统统一处理。

**对话处理流程（每轮）：**
1. 语义检索记忆 → 图遍历拉关联记忆 → 按 importance 排序
2. RAG 检索知识（按角色身份过滤）
3. 组装 system prompt：人格（含 few_shot）+ 状态 + RAG 知识 + 检索到的记忆
4. 组装 messages：最近 max_history 轮对话历史
5. 调用 LLM.chat() 生成回复
6. 状态机更新（关键词触发）
7. 记录记忆（LLM 抽取 → 存入 MemoryStore）
8. 更新对话历史

**对话历史管理（滑动窗口 + 摘要压缩）：**
- 保留最近 max_history 轮完整对话
- 当历史超过 max_history * 2 时，将超出部分送往 LLM 做一次摘要压缩（生成 2-3 句叙事摘要）
- 摘要结果作为 type=facts 的记忆存入 MemoryStore，然后截断历史到 max_history 轮
- 历史只包含 user/assistant 的消息列表，不重复注入 system prompt

**状态更新触发：**
- 使用分词级关键词匹配（非子串），每个触发词一组关键词
- betrayed: 背叛/欺骗/骗我/出卖
- helped: 帮忙/帮助/谢谢/救了我
- combat: 战斗/攻击/受伤/中招
- rested: 休息/睡觉/扎营/恢复

**记忆写入：**
- `_record_memory` 中 LLM 抽取失败时，改用简单规则记录（直接记录用户输入摘要），并至少打印日志提示
- 不静默吞掉所有异常

**配置初始化：**
- `GameMaster.__init__` 中必须调用 `self.knowledge.load_from_dir()` 加载知识文件
- 目录路径从 config.yaml 的 `knowledge.files` 读取
- ChromaDB 的 SentenceTransformer embedding function 应单例化（在 memory 和 knowledge 间共享，避免模型重复加载）

---

### 任务 11：CLI 入口 main.py

**涉及文件：**
- 新建：`trpg_agent/main.py`

**模块职责：** Rich CLI 入口，提供交互式跑团对话界面。

**功能：**
- 启动时显示欢迎面板（角色名 + 初始化状态）
- 每轮显示角色名 + 回复内容（Rich Panel）
- 支持 `/dice 3d6` 斜杠命令直接掷骰
- 支持 `exit`/`quit`/`退出` 退出
- ANTHROPIC_API_KEY 未设置时给出友好提示
- Ctrl+C 和 Ctrl+D 正常退出
- 初始化失败时显示错误原因

---

### 任务 12：集成验证与收尾

**步骤 1：运行全部确定性测试**

`pytest tests/ -v`，预期全部通过。

**步骤 2：运行冒烟测试**

验证所有模块可导入、角色加载成功、状态机工作、记忆/知识存储检索正常。

**步骤 3：最终提交**

---

## 验证方式

1. `pytest tests/ -v` — 全部确定性测试通过
2. 冒烟脚本确认所有模块可导入并正常工作
3. 手动运行 `python -m trpg_agent.main` 进行一轮对话验证

## 风险与注意事项

- sentence-transformers 首次运行需下载模型（~80MB），确保网络畅通
- ANTHROPIC_API_KEY 环境变量必须设置
- ChromaDB 数据在 `data/chroma/` 下分 memories/knowledge 两个子目录
- 记忆/知识测试必须使用 tmp_path，避免测试间污染
- 状态转换基于关键词匹配，实际跑团中可能需要按场景微调
- 对话历史摘要为异步优化（超出阈值时触发），首次达到前不会执行
