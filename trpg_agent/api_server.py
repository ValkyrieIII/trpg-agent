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

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from trpg_agent.game_master import GameMaster
from trpg_agent.event_stream import StatusEvent

# ------------------------------------------------------------------
#  Auth
# ------------------------------------------------------------------

_API_TOKEN = os.environ.get("TRPG_API_TOKEN", "")
_auth_scheme = HTTPBearer(auto_error=False)


def _require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_auth_scheme)) -> None:
    """If TRPG_API_TOKEN is set, require Bearer token match on all endpoints."""
    if not _API_TOKEN:
        return  # auth disabled — open access for local dev
    if credentials is None or credentials.credentials != _API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API token")


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
#  Health check
# ------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check — no auth required."""
    return {"status": "ok", "auth_enabled": bool(_API_TOKEN)}


# ------------------------------------------------------------------
#  Startup
# ------------------------------------------------------------------

@app.post("/api/start")
async def api_start(body: dict = {}, auth=Depends(_require_auth)):
    """开始新游戏。支持自定义世界观和NPC。"""
    try:
        # 开新游戏：销毁旧 GM 实例，清空记忆库，重建全新 GM
        global _gm
        if _gm is not None:
            _gm.memory.clear()
        _gm = None
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
            "suggestions": gm._last_suggestions or _extract_suggestions(opening),
            "state": gm.player_state.to_dict(),
            "npcs": _get_scene_npcs(gm),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/load")
async def api_load(auth=Depends(_require_auth)):
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
async def api_save(auth=Depends(_require_auth)):
    """保存游戏。"""
    try:
        gm = _get_gm()
        gm.save()
        return {"saved": True}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/status")
async def api_status(auth=Depends(_require_auth)):
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
async def api_npcs(auth=Depends(_require_auth)):
    """获取场景 NPC 列表。"""
    try:
        gm = _get_gm()
        return {"npcs": _get_scene_npcs(gm)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/knowledge")
async def api_knowledge(auth=Depends(_require_auth)):
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
async def api_command(body: dict, auth=Depends(_require_auth)):
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
async def api_debug(auth=Depends(_require_auth)):
    """获取 GM 调试日志并清空缓冲区。"""
    try:
        gm = _get_gm()
        logs = list(gm._debug_log)
        gm._debug_log.clear()
        return {"debug": gm.debug, "logs": logs}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/debug/toggle")
async def api_debug_toggle(auth=Depends(_require_auth)):
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
async def api_action(body: dict, auth=Depends(_require_auth)):
    """发送玩家行动，SSE 真流式返回 GM 响应。"""
    action = body.get("action", "")

    async def event_stream() -> AsyncGenerator[str, None]:
        gm = _get_gm()

        async for chunk in gm.process_streaming(action):
            if isinstance(chunk, StatusEvent):
                data = json.dumps({
                    'type': chunk.type,
                    'tool': chunk.tool,
                    'display': chunk.display,
                    'result': chunk.result,
                    'npc_name': chunk.npc_name,
                    'npc_text': chunk.npc_text,
                }, ensure_ascii=False)
            elif chunk:
                data = json.dumps({'text': chunk}, ensure_ascii=False)
            else:
                continue
            yield f"data: {data}\n\n"

        # After streaming: send final state from GameMaster
        response = gm._last_gm_response or ""
        final = {
            "done": True,
            "suggestions": gm._last_suggestions or _extract_suggestions(response),
            "state": gm.player_state.to_dict(),
            "npcs": _get_scene_npcs(gm),
            "gameOverPending": gm._game_over_pending,
            "gameOverCause": gm._game_over_cause if gm._game_over_pending else "",
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------
#  Game over confirmation endpoints
# ------------------------------------------------------------------


@app.post("/api/confirm_game_over")
async def api_confirm_game_over(auth=Depends(_require_auth)):
    """Confirm a pending game over — executes actual cleanup."""
    try:
        gm = _get_gm()
        result = gm._confirm_game_over()
        return {"result": result, "gameOver": gm._game_over}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/cancel_game_over")
async def api_cancel_game_over(auth=Depends(_require_auth)):
    """Cancel a pending game over — resumes gameplay."""
    try:
        gm = _get_gm()
        result = gm._cancel_game_over()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


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
