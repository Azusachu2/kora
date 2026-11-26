"""控制台用的只读管理接口：会话（Redis）、知识库与记忆（ChromaDB）、Agent 配置。

所有接口只读，不修改数据。启用鉴权时，这些路径不在普通用户可访问的列表里，只有管理员能调用。
"""
import asyncio
import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

_CONVERSATION_ID = re.compile(r"^[0-9a-f]{64}$")


def build_router(parts: Callable[[], Dict[str, Any]]) -> APIRouter:
    """parts() 在每次请求时返回当前组件，便于测试替换。"""
    router = APIRouter(prefix="/admin", tags=["控制台"])

    def ready(*names):
        found = parts()
        if any(found.get(name) is None for name in names):
            raise HTTPException(503, "服务未就绪")
        return found

    def knowledge_base(found):
        tool = getattr(found.get("tool_manager"), "_tools", {}).get("knowledge_search")
        return getattr(getattr(tool, "handler", None), "__self__", None)

    @router.get("/overview")
    async def overview():
        found = ready("memory", "orchestrator")
        memory, orchestrator = found["memory"], found["orchestrator"]
        kb = knowledge_base(found)
        conversations = [key async for key in memory._redis.scan_iter(match="wm_v2:*", count=500)]
        recognizer = getattr(orchestrator, "_intent_recognizer", None)
        return {
            "conversations": len(conversations),
            "knowledge_chunks": await asyncio.to_thread(kb._collection.count) if kb else 0,
            "episodic": await _count(memory, "_episodic"),
            "profiles": await _count(memory, "_profile"),
            "traces": await memory._redis.llen("trace:recent"),
            "models": {"agents": sorted({s.get("model") for s in orchestrator.get_stats().values() if s.get("model")}),
                       "intent": getattr(recognizer, "model", None),
                       "composer": getattr(getattr(orchestrator, "_composer", None), "_model", None)},
            "prefetch": getattr(orchestrator, "_knowledge_search", None) is not None,
        }

    @router.get("/agents")
    async def agents():
        orchestrator = ready("orchestrator")["orchestrator"]
        routing = {intent.value: agent.value for intent, agent in getattr(orchestrator, "_INTENT_ROUTING", {}).items()}
        return {"agents": orchestrator.get_stats(), "routing": routing}

    @router.get("/conversations")
    async def conversations(limit: int = Query(50, ge=1, le=200)):
        redis = ready("memory")["memory"]._redis
        items = []
        async for key in redis.scan_iter(match="wm_v2:*", count=500):
            conversation_id = key.split(":", 1)[1]
            latest, count, ttl = await asyncio.gather(redis.lindex(key, 0), redis.llen(key), redis.ttl(key))
            message = _message(latest) if latest else {}
            items.append({"id": conversation_id, "messages": count, "ttl_s": ttl, "last_ts": message.get("ts"),
                          "last_message": message.get("content", "")[:120]})
        items.sort(key=lambda item: item["last_ts"] or "", reverse=True)
        return {"items": items[:limit], "total": len(items)}

    @router.get("/conversations/{conversation_id}")
    async def conversation(conversation_id: str):
        if not _CONVERSATION_ID.match(conversation_id):
            raise HTTPException(400, "会话 ID 格式不正确")
        return await _conversation(ready("memory")["memory"], conversation_id)

    @router.get("/session")
    async def session(user_id: str = Query(..., min_length=1, max_length=200),
                      conv_id: str = Query(..., min_length=1, max_length=200)):
        """当前会话的完整状态：工作记忆、业务字段、摘要、用户画像和情景记忆。"""
        memory = ready("memory")["memory"]
        detail = await _conversation(memory, memory._conversation_id(user_id, conv_id))
        profile, episodic = await asyncio.gather(memory._get_profile(user_id),
                                                 _rows(memory, "_episodic", 20, 0, {"user_id": user_id}))
        return {**detail, "user_id": user_id, "conv_id": conv_id, "profile": profile, "episodic": episodic}

    @router.get("/knowledge")
    async def knowledge(q: Optional[str] = Query(None, max_length=2000), limit: int = Query(100, ge=1, le=500),
                        offset: int = Query(0, ge=0)):
        kb = knowledge_base(ready("tool_manager"))
        if kb is None:
            raise HTTPException(503, "知识库未初始化")
        if q and q.strip():
            items = await kb.search_async(q.strip(), top_k=min(limit, 20))
            return {"items": items, "total": len(items), "query": q}
        total = await asyncio.to_thread(kb._collection.count)
        rows = await asyncio.to_thread(kb._collection.get, limit=limit, offset=offset, include=["documents", "metadatas"])
        items = [{"id": chunk_id, "title": (meta or {}).get("title", ""), "document_id": (meta or {}).get("document_id", ""),
                  "chunk": (meta or {}).get("chunk_index", 0), "total_chunks": (meta or {}).get("total_chunks", 1),
                  "version": (meta or {}).get("version", ""), "content": document}
                 for chunk_id, document, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])]
        items.sort(key=lambda item: (item["title"], item["chunk"]))
        return {"items": items, "total": total}

    @router.get("/memory/episodic")
    async def episodic(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        memory = ready("memory")["memory"]
        return {"items": await _rows(memory, "_episodic", limit, offset), "total": await _count(memory, "_episodic")}

    @router.get("/memory/profiles")
    async def profiles(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        memory = ready("memory")["memory"]
        return {"items": await _rows(memory, "_profile", limit, offset), "total": await _count(memory, "_profile")}

    return router


def _message(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return {"role": data.get("role"), "content": data.get("content", ""), "ts": data.get("ts"),
            "client_request_id": (data.get("metadata") or {}).get("request_id")}


async def _conversation(memory, conversation_id: str) -> Dict[str, Any]:
    redis = memory._redis
    raws, summary, state = await asyncio.gather(redis.lrange(f"wm_v2:{conversation_id}", 0, -1),
                                                redis.get(f"summary_v2:{conversation_id}"),
                                                redis.get(f"state_v1:{conversation_id}"))
    messages = [_message(raw) for raw in reversed(raws)]          # Redis 里最新的在前
    turns: List[Dict[str, Any]] = []
    for message in messages:
        client_id = message.get("client_request_id")
        if message["role"] == "user" or not turns or turns[-1].get("client_request_id") != client_id:
            turns.append({"client_request_id": client_id, "user": None, "assistant": None, "ts": message.get("ts")})
        turns[-1]["user" if message["role"] == "user" else "assistant"] = message.get("content")
    receipts = await asyncio.gather(*(redis.get(f"turn_receipt:{conversation_id}:"
                                                f"{hashlib.sha256(turn['client_request_id'].encode()).hexdigest()}")
                                      if turn.get("client_request_id") else _none() for turn in turns))
    for turn, receipt in zip(turns, receipts):
        response = (json.loads(receipt) if receipt else {}).get("response") or {}
        turn["response"] = {key: response.get(key) for key in (
            "request_id", "intent", "intent_group", "intent_confidence", "primary_agent", "supporting_agents", "tools_used",
            "knowledge_used", "escalated", "latency_ms", "status", "entities", "performance")} if response else None
    try:
        state_data = json.loads(state) if state else {}
    except ValueError:
        state_data = {}
    return {"id": conversation_id, "messages": messages, "turns": turns, "summary": summary or "", "state": state_data}


async def _none():
    return None


async def _count(memory, name: str) -> int:
    collection = getattr(memory, name, None)
    if collection is None:
        return 0
    try:
        return await asyncio.to_thread(collection.count)
    except Exception:
        return 0


async def _rows(memory, name: str, limit: int, offset: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    collection = getattr(memory, name, None)
    if collection is None:
        return []
    kwargs = {"limit": limit, "offset": offset, "include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where
    try:
        rows = await asyncio.to_thread(collection.get, **kwargs)
    except Exception:
        return []
    items = []
    for row_id, document, meta in zip(rows.get("ids", []), rows.get("documents", []), rows.get("metadatas", [])):
        meta = meta or {}
        try:
            parsed = json.loads(document) if name == "_profile" else document
        except (TypeError, ValueError):
            parsed = document
        items.append({"id": row_id, "user_id": meta.get("user_id"), "conv_id": meta.get("conv_id"),
                      "time": meta.get("updated_at") or meta.get("ts"), "content": parsed,
                      "full_text": meta.get("full_text")})
    items.sort(key=lambda item: item["time"] or "", reverse=True)
    return items
