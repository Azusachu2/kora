"""控制台：只读管理接口、轨迹时间线、鉴权和页面。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

import api.main as api
from core.performance import collect_performance, measure_stage, offset_ms, performance_snapshot, record_llm_call
from test_chat_reliability import body, services  # noqa: F401


class FakeKB:
    def __init__(self):
        rows = {"ids": ["d1:0", "d1:1", "d2:0"], "documents": ["退款原路退回", "信用卡体现为退款记录", "配送 3-5 天"],
                "metadatas": [{"title": "退款到账说明", "document_id": "d1", "chunk_index": 0, "total_chunks": 2, "version": "v"},
                              {"title": "退款到账说明", "document_id": "d1", "chunk_index": 1, "total_chunks": 2, "version": "v"},
                              {"title": "配送说明", "document_id": "d2", "chunk_index": 0, "total_chunks": 1, "version": "v"}]}
        self.calls = []
        self._collection = SimpleNamespace(count=lambda: 3, get=lambda **kwargs: self.calls.append(kwargs) or rows)

    async def search_async(self, query, top_k=5):
        return [{"title": "退款到账说明", "content": "退款原路退回", "score": 0.71}]

    async def search_handler(self, params, context):
        return []


def with_knowledge(monkeypatch):
    kb = FakeKB()
    monkeypatch.setattr(api, "_tool_manager", SimpleNamespace(_tools={"knowledge_search": SimpleNamespace(handler=kb.search_handler)}))
    return kb


@pytest.mark.asyncio
async def test_session_view_matches_what_the_chat_stored(services, monkeypatch):  # noqa: F811
    memory, orch, _, http = services
    with_knowledge(monkeypatch)
    reply = (await http.post("/chat", json=body("订单 A12345 退款500元"))).json()
    session = (await http.get("/admin/session", params={"user_id": "u", "conv_id": "c"})).json()
    assert [m["role"] for m in session["messages"]] == ["user", "assistant"]
    turn = session["turns"][0]
    assert turn["user"] == "订单 A12345 退款500元" and turn["response"]["request_id"] == reply["request_id"]
    assert session["state"]["entities"]["amount"] and session["profile"] == {} and session["episodic"] == []
    listed = (await http.get("/admin/conversations")).json()
    assert listed["total"] == 1 and listed["items"][0]["id"] == session["id"] and listed["items"][0]["messages"] == 2
    assert (await http.get(f"/admin/conversations/{session['id']}")).json()["turns"] == session["turns"]
    assert (await http.get("/admin/conversations/not-a-hash")).status_code == 400
    overview = (await http.get("/admin/overview")).json()
    assert overview["conversations"] == 1 and overview["knowledge_chunks"] == 3 and overview["traces"] == 1


@pytest.mark.asyncio
async def test_stored_trace_has_a_timeline_without_conversation_text(services):  # noqa: F811
    memory, orch, _, http = services
    reply = (await http.post("/chat", json=body("退款规则"))).json()
    trace = (await http.get(f"/trace/tool/{reply['request_id']}")).json()["trace"]
    assert {"memory_read", "intent", "agent_execution"} <= {stage["name"] for stage in trace["timeline"]}
    assert all(stage["start_ms"] >= 0 for stage in trace["timeline"])
    assert trace["llm_calls"][0]["stage"].startswith("agent.") and trace["llm_calls"][0]["start_ms"] >= 0
    assert trace["routing_reason"] and trace["total_ms"] > 0
    assert "退款规则" not in json.dumps(trace, ensure_ascii=False)


@pytest.mark.asyncio
async def test_knowledge_listing_and_semantic_search(services, monkeypatch):  # noqa: F811
    _, _, _, http = services
    kb = with_knowledge(monkeypatch)
    listing = (await http.get("/admin/knowledge", params={"limit": 2, "offset": 1})).json()
    assert listing["total"] == 3 and kb.calls[-1]["limit"] == 2 and kb.calls[-1]["offset"] == 1
    assert [(i["title"], i["chunk"]) for i in listing["items"]] == [("退款到账说明", 0), ("退款到账说明", 1), ("配送说明", 0)]
    found = (await http.get("/admin/knowledge", params={"q": "多久到账"})).json()
    assert found["items"][0]["score"] == 0.71 and found["query"] == "多久到账"


@pytest.mark.asyncio
async def test_admin_data_needs_an_admin_token_but_the_page_does_not(services, monkeypatch):  # noqa: F811
    _, _, _, http = services
    tokens = {"reader-token-for-console-tests": {"user_id": "r", "role": "user"},
              "admin-token-for-console-tests!": {"user_id": "a", "role": "admin"}}
    monkeypatch.setenv("KORA_API_TOKENS", json.dumps(tokens))
    reader, admin = ({"Authorization": f"Bearer {token}"} for token in tokens)
    assert (await http.get("/admin/overview")).status_code == 401
    assert (await http.get("/admin/overview", headers=reader)).status_code == 403
    assert (await http.get("/admin/conversations", headers=admin)).status_code == 200
    page = await http.get("/console/")
    assert page.status_code == 200 and "Kora Console" in page.text
    assert "http://" not in page.text and "https://" not in page.text          # 不依赖外部资源


@pytest.mark.asyncio
async def test_not_ready_is_503(monkeypatch):
    monkeypatch.setattr(api, "_memory", None)
    import httpx
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
        assert (await http.get("/admin/conversations")).status_code == 503


@pytest.mark.asyncio
async def test_timeline_offsets_are_relative_to_the_turn():
    assert offset_ms() is None
    with collect_performance():
        await asyncio.sleep(0.01)
        with measure_stage("intent"):
            await asyncio.sleep(0.01)
        record_llm_call("intent", "m", __import__("time").monotonic(), "success")
        snapshot = performance_snapshot()
    stage = snapshot["timeline"][0]
    assert stage["name"] == "intent" and stage["start_ms"] >= 9 and stage["ms"] >= 9
    assert snapshot["llm_calls"][0]["start_ms"] >= stage["start_ms"] and "_t0" not in snapshot
