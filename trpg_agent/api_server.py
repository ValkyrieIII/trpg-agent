"""FastAPI 后端 — 为 TRPG Web 前端提供 API + SSE 流式输出。

启动方式:
    python -m trpg_agent.api_server

或:
    uvicorn trpg_agent.api_server:app --reload --port 8000
"""

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

# 显式禁用 ChromaDB 遥测（必须在 import chromadb 之前设置）
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from trpg_agent.game_master import GameMaster

# ------------------------------------------------------------------
#  App
# ------------------------------------------------------------------

app = FastAPI(title="TRPG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
#  Global GameMaster instance (lazy init)
# ------------------------------------------------------------------

_gm: GameMaster | None = None
_config_path = os.environ.get("TRPG_CONFIG", "config.yaml")


def _get_gm() -> GameMaster:
    """Get or create the global GameMaster instance."""
    global _gm
    if _gm is not None:
        return _gm

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    debug = os.environ.get("TRPG_DEBUG", "").lower() in ("true", "1", "yes")
    _gm = GameMaster(
        config_path=_config_path,
        llm_api_key=api_key,
        debug=debug,
    )
    return _gm


# ------------------------------------------------------------------
#  Startup
# ------------------------------------------------------------------

@app.post("/api/start")
async def api_start(body: dict = {}):
    """开始新游戏。支持自定义世界观和NPC。"""
    try:
        gm = _get_gm()

        # 接收前端传入的自定义世界观和NPC
        custom_worldview = body.get("worldview", "")
        custom_npc_setup = body.get("npc_setup", "")

        # 如果有自定义世界观，更新 GM 的世界状态，影响后续所有游戏流程
        if custom_worldview.strip():
            gm.world["description"] = custom_worldview.strip()
            # 从描述中提取第一句作为世界名称（简化处理）
            first_line = custom_worldview.strip().split("。")[0]
            if first_line:
                gm.world["name"] = first_line

        opening = gm.generate_opening(
            custom_worldview=custom_worldview,
            custom_npc_setup=custom_npc_setup,
        )
        return {
            "opening": opening,
            "suggestions": _extract_suggestions(opening),
            "state": gm.player_state.to_dict(),
            "npcs": _get_scene_npcs(gm),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/load")
async def api_load():
    """加载存档。"""
    try:
        gm = _get_gm()
        loaded = gm.load_save()
        if not loaded:
            return {"error": "没有找到存档"}
        return {
            "loaded": True,
            "state": gm.player_state.to_dict(),
            "npcs": _get_scene_npcs(gm),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/save")
async def api_save():
    """保存游戏。"""
    try:
        gm = _get_gm()
        gm.save()
        return {"saved": True}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/status")
async def api_status():
    """获取玩家状态。"""
    try:
        gm = _get_gm()
        return {
            "state": gm.player_state.to_dict(),
            "npcs": _get_scene_npcs(gm),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/npcs")
async def api_npcs():
    """获取场景 NPC 列表。"""
    try:
        gm = _get_gm()
        return {"npcs": _get_scene_npcs(gm)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/knowledge")
async def api_knowledge():
    """获取知识库条目（简化版，返回最近添加的条目）。"""
    try:
        gm = _get_gm()
        results = gm.knowledge.search("", top_k=50)
        entries = []
        for r in results:
            meta = r.get("metadata", {})
            entries.append({
                "content": r.get("content", ""),
                "category": meta.get("category", "未分类"),
                "known_by": meta.get("known_by", "所有人"),
            })
        return {"knowledge": entries}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/command")
async def api_command(body: dict):
    """执行 ! 命令（如 !npc, !world）。"""
    try:
        gm = _get_gm()
        command = body.get("command", "")
        result = gm._handle_world_builder_command(command)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
#  Debug endpoint
# ------------------------------------------------------------------

@app.get("/api/debug")
async def api_debug():
    """获取 GM 调试日志并清空缓冲区。"""
    try:
        gm = _get_gm()
        logs = list(gm._debug_log)
        gm._debug_log.clear()
        return {"debug": gm.debug, "logs": logs}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/debug/toggle")
async def api_debug_toggle():
    """切换调试模式。"""
    try:
        gm = _get_gm()
        gm.debug = not gm.debug
        return {"debug": gm.debug}
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
#  SSE Action endpoint
# ------------------------------------------------------------------

@app.post("/api/action")
async def api_action(body: dict):
    """发送玩家行动，SSE 流式返回 GM 响应。"""
    action = body.get("action", "")

    async def event_stream() -> AsyncGenerator[str, None]:
        gm = _get_gm()

        # 调用 process 获取结果
        result = gm.process(action)

        # Send narration in chunks for streaming effect
        if result:
            sentences = _split_sentences(result)
            for sentence in sentences:
                chunk = json.dumps({'text': sentence}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.05)

        # Send final data — no narration field to avoid duplication on frontend
        final = {
            "done": True,
            "suggestions": _extract_suggestions(result),
            "state": gm.player_state.to_dict(),
            "npcs": _get_scene_npcs(gm),
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _get_scene_npcs(gm: GameMaster) -> list[dict]:
    """Get scene NPCs with their states."""
    npcs = []
    for name in gm.scene_npcs:
        npc = gm.npc_store.find_by_name(name)
        state = gm.npc_store.get_state(name)
        state_dict = state.get_state() if state else {}
        npcs.append({
            "name": name,
            "emotion": state_dict.get("emotion", "calm"),
            "trust": state_dict.get("trust", 0.5),
            "hp": state_dict.get("hp", 10),
            "max_hp": state_dict.get("max_hp", 10),
        })
    return npcs


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming effect."""
    import re
    # Split by Chinese/English sentence endings
    parts = re.split(r'([。！？；\n]+)', text)
    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]
        if sentence.strip():
            sentences.append(sentence)
    return sentences


def _extract_suggestions(text: str) -> list[str]:
    """Extract numbered suggestions from text."""
    import re
    suggestions = []
    for match in re.finditer(r'^\d+\.\s+(.+)$', text, re.MULTILINE):
        suggestions.append(match.group(1))
    return suggestions


# ------------------------------------------------------------------
#  Static files (built frontend)
# ------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent.parent / "web" / "dist"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


# ------------------------------------------------------------------
#  Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
