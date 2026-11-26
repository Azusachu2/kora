"""知识库预取、辅助模型和记忆并行读取。"""
import asyncio
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import api.main as api
from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, BillingAgent, GeneralAgent, Request, ResponseComposer, TechnicalAgent,
)
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from core.llm_utils import create_message, fast_model
from core.streaming import Emitter
from evaluation.evaluator import EndToEndEvaluator, QualityScores
from memory.conversation_memory import MemoryManager
from test_chat_reliability import body, services  # noqa: F401
from tooling.tool_manager import MCPToolManager, ToolResult


def item(title, content="正文", score=0.6):
    return {"title": title, "content": content, "score": score}


def orchestrator(search=None, reply="回复"):
    create = AsyncMock(return_value=SimpleNamespace(content=[{"type": "text", "text": reply}], stop_reason="end_turn"))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {kind: [cls(client, "m")] for kind, cls in
                  ((AgentType.GENERAL, GeneralAgent), (AgentType.BILLING, BillingAgent), (AgentType.TECHNICAL, TechnicalAgent))}
    orch._composer = ResponseComposer(client, "m")
    orch._recent_tool_traces = deque(maxlen=10)
    orch.set_shared_tools(build_shared_rag_tools(None))
    if search:
        orch.set_knowledge_prefetch(search)
    return orch, create


def request(message="退款审核通过后多久可以到账", intent=IntentCategory.REFUND, **kwargs):
    return Request(message, "u", "c", intent=intent, intent_group="billing", urgency=kwargs.pop("urgency", UrgencyLevel.LOW),
                   intent_confidence=0.95, **kwargs)


def recording_search(*results):
    queries, queue = [], list(results)

    async def search(query):
        queries.append(query)
        return queue.pop(0) if len(queue) > 1 else queue[0]
    return search, queries


def prompt_text(create, call=0):
    return "\n".join(str(m["content"]) for m in create.call_args_list[call].kwargs["messages"])


# ── 预取 ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prefetched_knowledge_reaches_the_first_model_call():
    search, queries = recording_search(ToolResult(True, [item("退款到账说明", "三到五个工作日")], "knowledge_search"))
    orch, create = orchestrator(search)
    result = await orch.run(request())
    assert create.await_count == 1 and queries == ["退款审核通过后多久可以到账"]
    text = prompt_text(create)
    assert "《退款到账说明》\n三到五个工作日" in text and "0.6" not in text        # 不给模型看检索分数
    roles = [m["role"] for m in create.call_args.kwargs["messages"]]
    assert all(a != b for a, b in zip(roles, roles[1:]))                        # 角色交替
    trace = result.tool_traces[0]
    assert trace["event"] == "knowledge_prefetch" and trace["result_success"] and result.tools_used == []
    assert trace["result_titles"] == ["退款到账说明"]          # 只记标题，供控制台标出本轮检索到的文档


@pytest.mark.asyncio
@pytest.mark.parametrize("intent,urgency", [
    (IntentCategory.GREETING, UrgencyLevel.LOW), (IntentCategory.FEEDBACK, UrgencyLevel.LOW),
    (IntentCategory.HUMAN_HANDOFF, UrgencyLevel.LOW), (IntentCategory.REFUND, UrgencyLevel.CRITICAL),
])
async def test_no_prefetch_when_knowledge_is_not_needed(intent, urgency):
    search, queries = recording_search(ToolResult(True, [item("a")], "knowledge_search"))
    orch, create = orchestrator(search)
    await orch.run(request("你们的服务真不错", intent, urgency=urgency))
    assert queries == []
    if intent in (IntentCategory.GREETING, IntentCategory.FEEDBACK):
        kwargs = create.call_args.kwargs
        assert "tools" not in kwargs and "search_knowledge_base" not in kwargs["system"]


@pytest.mark.asyncio
async def test_clarification_does_not_search():
    search, queries = recording_search(ToolResult(True, [], "knowledge_search"))
    orch, create = orchestrator(search)
    req = request("那个东西怎么弄", IntentCategory.OTHER)
    req.intent_confidence = 0.2
    await orch.run(req)
    assert queries == [] and create.await_count == 0


@pytest.mark.asyncio
async def test_empty_result_is_stated_and_failure_falls_back_to_the_tool():
    search, _ = recording_search(ToolResult(True, [], "knowledge_search"))
    orch, create = orchestrator(search)
    result = await orch.run(request())
    assert "没有找到相关资料" in prompt_text(create) and not result.tool_traces[0]["result_success"]

    async def hang(query):
        await asyncio.Event().wait()

    async def broken(query):
        raise RuntimeError("连接断开")
    failed, _ = recording_search(ToolResult(False, [item("降级结果")], "knowledge_search", error="熔断中"))
    for search in (failed, broken, hang):
        orch, create = orchestrator(search)
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("KORA_PREFETCH_TIMEOUT_S", "0.1")
            result = await asyncio.wait_for(orch.run(request()), 5)
        assert result.status == "success" and "[知识库资料]" not in prompt_text(create)
        assert result.tool_traces[0]["error"] and not result.tool_traces[0]["result_success"]
        assert "search_knowledge_base" in {t["name"] for t in create.call_args.kwargs["tools"]}


@pytest.mark.asyncio
async def test_without_prefetch_behaviour_is_unchanged():
    orch, create = orchestrator()
    result = await orch.run(request())
    assert result.tool_traces == [] and "[知识库资料]" not in prompt_text(create)


@pytest.mark.parametrize("message,history,expected", [
    ("那多久到账", [{"role": "user", "content": "退款审核通过了"}, {"role": "assistant", "content": "好的"}], ["那多久到账", "退款审核通过了 那多久到账"]),
    ("那多久到账", None, ["那多久到账"]),
    ("退款审核通过之后一般要多久才能到账", [{"role": "user", "content": "前文"}], ["退款审核通过之后一般要多久才能到账"]),
    ("how long does it take?", [{"role": "user", "content": "My refund was approved yesterday"}],
     ["how long does it take?", "My refund was approved yesterday how long does it take?"]),
    ("How long does it take to receive my refund?", [{"role": "user", "content": "I returned the headphones"}],
     ["How long does it take to receive my refund?"]),
])
def test_short_follow_ups_search_with_and_without_context(message, history, expected):
    assert AgentOrchestrator._knowledge_queries(Request(message, "u", "c", history=history)) == expected


def test_merge_alternates_removes_duplicates_and_keeps_single_source_unchanged():
    a, b = [item(f"a{i}") for i in range(5)], [item("a0"), item("b1"), item("b2"), item("b3")]
    assert [i["title"] for i in AgentOrchestrator._merge_knowledge([a])] == ["a0", "a1", "a2", "a3", "a4"]
    assert [i["title"] for i in AgentOrchestrator._merge_knowledge([a, b])] == ["a0", "a1", "b1", "a2", "b2", "a3", "b3"]
    assert AgentOrchestrator._merge_knowledge([]) == []


@pytest.mark.asyncio
async def test_parallel_agents_share_one_prefetch():
    search, queries = recording_search(ToolResult(True, [item("登录失败", "重置密码")], "knowledge_search"))
    orch, create = orchestrator(search)
    result = await orch.run(request("登录失败报错401还被重复扣款", IntentCategory.PAYMENT_ISSUE))
    assert len(queries) == 1 and len(result.agent_types) == 2
    assert all("《登录失败》" in prompt_text(create, n) for n in range(2))
    assert sum(t.get("event") == "knowledge_prefetch" for t in result.tool_traces) == 1


@pytest.mark.asyncio
async def test_chat_marks_knowledge_used_and_reports_stages(services):  # noqa: F811
    _, orch, _, http = services
    search, _ = recording_search(ToolResult(True, [item("退款政策")], "knowledge_search"))
    orch.set_knowledge_prefetch(search)
    reply = (await http.post("/chat", json=body("退款审核通过后多久可以到账"))).json()
    assert reply["knowledge_used"] and "knowledge_prefetch" in reply["performance"]["stages_ms"]
    emitter = Emitter()
    await orch.run(request(emit=None))
    req = request()
    req.emit = emitter
    await orch._prefetch_knowledge(req)
    assert emitter._queue.get_nowait() == ("status", {"stage": "knowledge"})


@pytest.mark.asyncio
async def test_prefetch_never_calls_the_model_even_with_rewrite_enabled():
    manager = MCPToolManager(api_key="x", model="m")
    manager.rag_rewrite = manager.rag_rerank = True
    manager.rewrite_query = AsyncMock(side_effect=AssertionError("不应改写"))
    manager.call = AsyncMock(return_value=ToolResult(True, [], "knowledge_search"))
    assert (await api._knowledge_prefetch(manager)("退款")).success
    manager.call.assert_awaited_once_with("knowledge_search", {"query": "退款", "top_k": 5})
    await manager._client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("found", [True, False])
async def test_evaluation_counts_prefetched_knowledge_as_tool_use(found):
    search, _ = recording_search(ToolResult(True, [item("退款政策")] if found else [], "knowledge_search"))
    orch, _ = orchestrator(search)
    orch._intent_recognizer = SimpleNamespace(recognize=AsyncMock(return_value=IntentResult(
        IntentCategory.REFUND, .95, UrgencyLevel.LOW, "billing", {}, "test", 0)))
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._orchestrator = orch
    evaluator._judge = SimpleNamespace(judge=AsyncMock(return_value=QualityScores(1, 1, 1, 1)))
    results = await evaluator._evaluate_dialog_case(
        {"question": "退款审核通过后多久可以到账", "required_tools": ["search_knowledge_base"]}, 0)
    assert results[0].passed is found


# ── 辅助模型 ──────────────────────────────────────────────────────────────────

def test_fast_model_is_optional(monkeypatch):
    monkeypatch.delenv("KORA_FAST_MODEL", raising=False)
    assert fast_model("main") == "main"
    monkeypatch.setenv("KORA_FAST_MODEL", "  small  ")
    assert fast_model("main") == "small"


def test_only_intent_recognition_moves_to_the_fast_model(monkeypatch):
    monkeypatch.setenv("KORA_FAST_MODEL", "small")
    orch = AgentOrchestrator(api_key="x", model="main")
    assert orch._intent_recognizer.model == "small" and orch._composer._model == "main"
    assert {agent._model for agents in orch._pool.values() for agent in agents} == {"main"}


@pytest.mark.asyncio
@pytest.mark.parametrize("model,kept", [
    ("claude-opus-5-5", False), ("claude-haiku-5-5", False), ("claude-sonnet-5-5", False), ("claude-fable-5-2", False),
    ("claude-3-5-sonnet-20241022", True), ("deepseek-v4-pro", True),
])
async def test_temperature_and_effort_follow_the_model(model, kept):
    create = AsyncMock()
    client = SimpleNamespace(base_url="", messages=SimpleNamespace(create=create))
    kwargs = {"model": model, "temperature": 0.0, "max_tokens": 9}
    await create_message(client, effort="low", **kwargs)
    sent = create.call_args.kwargs
    assert ("temperature" in sent) is kept and ("extra_body" not in sent) is kept
    assert kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_background_memory_calls_use_low_effort():
    create = AsyncMock(return_value=SimpleNamespace(content=[{"type": "text", "text": "摘要"}], stop_reason="end_turn"))
    memory = MemoryManager.__new__(MemoryManager)
    memory._client, memory._model = SimpleNamespace(base_url="", messages=SimpleNamespace(create=create)), "claude-haiku-5-5"
    assert await memory._merge_summary("旧摘要", "新摘要") == "摘要"
    sent = create.call_args.kwargs
    assert sent["extra_body"] == {"output_config": {"effort": "low"}} and sent["max_tokens"] == 1024


# ── 记忆并行读取 ──────────────────────────────────────────────────────────────

def slow_memory(delay=0.1):
    memory = MemoryManager.__new__(MemoryManager)
    seen = {}

    async def wait(value):
        await asyncio.sleep(delay)
        return value

    async def episodic(user_id, conv_id, query):
        seen["query"] = query
        return await wait(["历史"])
    memory._get_working_memory = lambda *a: wait([SimpleNamespace(content="最近一条")])
    memory._search_episodic = episodic
    memory._get_profile = lambda *a: wait({"偏好": "短"})
    memory._redis = SimpleNamespace(get=lambda key: wait('{"entities": {}}' if key.startswith("state") else "摘要"))
    return memory, seen


@pytest.mark.asyncio
async def test_context_is_read_in_parallel_and_complete():
    memory, seen = slow_memory()
    started = time.monotonic()
    context = await memory.get_context("u", "c", query="退款")
    assert time.monotonic() - started < 0.25        # 串行需要 0.5 秒
    assert (context.relevant_history, context.user_profile, context.summary, context.state) == (["历史"], {"偏好": "短"}, "摘要", {"entities": {}})
    assert seen["query"] == "退款"
    await memory.get_context("u", "c")
    assert seen["query"] == "最近一条"
