"""流式输出：SDK 边界、工具循环、失败兜底和 HTTP 接口。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import AsyncAnthropic

import api.main as api
from agents.agent_orchestrator import (
    AgentResponse, AgentType, BillingAgent, EscalationAgent, GeneralAgent, Request, ResponseComposer, RoutingDecision,
)
from agents.tools import make_tool
from core import streaming
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from core.llm_utils import StreamInterrupted, stream_message
from core.performance import collect_performance, performance_snapshot
from core.streaming import Emitter
from test_chat_reliability import body, services  # noqa: F401


# ── 假的模型流 ────────────────────────────────────────────────────────────────

def message_events(*blocks, stop="end_turn", finish=True, model="m"):
    """每个块：("text", 片段...)、("thinking", 签名) 或 ("tool", 名称, 入参)。"""
    events = [{"type": "message_start", "message": {"id": "x", "type": "message", "role": "assistant", "model": model, "content": [],
                                                    "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 1}}}]
    for index, (kind, *rest) in enumerate(blocks):
        if kind == "text":
            start, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": piece} for piece in rest]
        elif kind == "thinking":
            start, deltas = {"type": "thinking", "thinking": "", "signature": ""}, [{"type": "signature_delta", "signature": rest[0]}]
        else:
            raw = json.dumps(rest[1], ensure_ascii=False)
            start = {"type": "tool_use", "id": "t1", "name": rest[0], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": part} for part in (raw[:4], raw[4:])]
        events.append({"type": "content_block_start", "index": index, "content_block": start})
        events += [{"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas]
        events.append({"type": "content_block_stop", "index": index})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 33}})
    return events + ([{"type": "message_stop"}] if finish else [])


def sse_response(events):
    text = "".join(f"event: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=text.encode())


def http_client(handler):
    return AsyncAnthropic(api_key="test-only", base_url="https://api.anthropic.com", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class FakeStream:
    def __init__(self, events, error=None):
        self.events, self.error, self.closed = events, error, False

    async def __aiter__(self):
        for event in self.events:
            await asyncio.sleep(0)
            yield event
        if self.error:
            raise self.error

    async def close(self):
        self.closed = True


def streaming_client(*replies):
    """依次返回各次调用的结果；异常实例直接抛出。"""
    calls, queue = [], list(replies)

    async def create(**kwargs):
        calls.append(kwargs)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, FakeStream) else FakeStream(reply)
    return SimpleNamespace(messages=SimpleNamespace(create=create), calls=calls)


def drain(emitter):
    """已发出的事件；最后按客户端的规则拼出用户看到的正文。"""
    events = []
    while not emitter._queue.empty():
        events.append(emitter._queue.get_nowait())
    return events, visible(events)


def visible(events):
    text = ""
    for event, data in events:
        text = text + data["text"] if event == "delta" else "" if event == "reset" else text
    return text


async def run_agent(agent, emitter, **kwargs):
    return await agent.handle(Request(message="退款多久到账", user_id="u", conv_id="c", emit=emitter, **kwargs))


# ── SDK 边界 ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_keeps_thinking_signature_tool_input_and_usage():
    sent, seen = [], []

    def handle(request):
        sent.append(json.loads(request.content))
        return sse_response(message_events(("thinking", "SIG123"), ("text", "我先", "查一下"), ("text",),
                                           ("tool", "search", {"query": "退款"}), stop="tool_use", model="claude-opus-5-5"))
    with collect_performance():
        async with http_client(handle) as client:
            message = await stream_message(client, on_text=seen.append, stage="s", effort="low", model="claude-opus-5-5",
                                           temperature=0.2, max_tokens=256, messages=[{"role": "user", "content": "hi"}])
        row = performance_snapshot()["llm_calls"][0]
    assert sent[0]["stream"] is True and "temperature" not in sent[0] and sent[0]["output_config"] == {"effort": "low"}
    assert message.content == [
        {"type": "thinking", "thinking": "", "signature": "SIG123"}, {"type": "text", "text": "我先查一下"},
        {"type": "tool_use", "id": "t1", "name": "search", "input": {"query": "退款"}}]      # 空正文块已去掉
    assert message.stop_reason == "tool_use" and message.model == "claude-opus-5-5"
    assert "".join(seen).strip() == "我先查一下"
    assert row["status"] == "success" and (row["input_tokens"], row["output_tokens"]) == (12, 33)


@pytest.mark.asyncio
async def test_interrupted_and_failed_streams_are_errors_not_answers():
    cut = lambda request: sse_response(message_events(("text", "退款需要"), finish=False))
    overloaded = lambda request: sse_response(message_events(("text", "退款"), finish=False)[:3] + [
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}])
    for handler, expected in ((cut, StreamInterrupted), (overloaded, Exception)):
        with collect_performance():
            async with http_client(handler) as client:
                with pytest.raises(expected):
                    await stream_message(client, on_text=lambda text: None, model="m", max_tokens=9, messages=[])
            assert performance_snapshot()["llm_calls"][0]["status"] == "error"


@pytest.mark.asyncio
async def test_stream_is_closed_when_the_caller_times_out():
    class Hanging(FakeStream):
        async def __aiter__(self):
            yield message_events(("text", "a"))[0]
            await asyncio.sleep(60)
    stream = Hanging([])
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(stream_message(streaming_client(stream), on_text=lambda text: None, model="m"), 0.05)
    assert stream.closed


# ── Agent 与编排 ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_text_before_a_tool_call_is_discarded_and_signature_is_sent_back():
    client = streaming_client(message_events(("thinking", "SIG"), ("text", "我先查一下"), ("tool", "lookup", {"q": "a"}), stop="tool_use"),
                              message_events(("text", "审核通过后", "三到五天到账")))
    agent = GeneralAgent(client, "m")
    agent.set_shared_tools({"lookup": make_tool("lookup", "d", {"q": {"type": "string"}}, lambda req, args: {"success": True})})
    emitter = Emitter()
    response = await run_agent(agent, emitter)
    events, text = drain(emitter)
    assert response.success and text == response.content == "审核通过后三到五天到账"
    assert [e for e, _ in events] == ["delta", "reset", "status", "delta", "delta"]
    assistant = client.calls[1]["messages"][-2]
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "SIG"}


@pytest.mark.asyncio
async def test_failed_specialist_text_is_discarded_before_fallback(services):  # noqa: F811
    _, orch, _, _ = services
    broken = streaming_client(FakeStream(message_events(("text", "半截回复"), finish=False), RuntimeError("断开")))
    healthy = streaming_client(message_events(("text", "兜底回复")))
    orch._pool[AgentType.BILLING] = [BillingAgent(broken, "m")]
    orch._pool[AgentType.GENERAL] = [GeneralAgent(healthy, "m")]
    emitter = Emitter()
    response = await orch._execute(Request("退款", "u", "c", emit=emitter), AgentType.BILLING)
    events, text = drain(emitter)
    assert response.content == text == "兜底回复"
    assert ("reset", {}) in events and visible(events[:2]) == ""


@pytest.mark.asyncio
async def test_parallel_agents_stream_only_the_composed_reply(services):  # noqa: F811
    _, orch, _, _ = services
    seen = []

    async def execute(req, agent_type):
        seen.append(req.emit)
        return AgentResponse(agent_type=agent_type, content=agent_type.value, success=True)
    orch._execute = execute
    orch._composer = ResponseComposer(streaming_client(message_events(("text", "汇总", "回复"))), "m")
    emitter = Emitter()
    req = Request("登录失败还被扣款", "u", "c", intent=IntentCategory.PAYMENT_ISSUE, emit=emitter)
    result = await orch.run_parallel(req, RoutingDecision(AgentType.BILLING, [AgentType.TECHNICAL]))
    assert seen == [None, None] and drain(emitter)[1] == result.response == "汇总回复"


@pytest.mark.asyncio
async def test_composer_failure_replaces_partial_text(services):  # noqa: F811
    _, orch, _, _ = services
    orch._composer = ResponseComposer(streaming_client(FakeStream(message_events(("text", "汇总到一半"), finish=False))), "m")
    responses = [AgentResponse(kind, f"{kind.value} 结论", True) for kind in (AgentType.BILLING, AgentType.TECHNICAL)]
    emitter = Emitter()
    result = await orch._composer.compose_result(Request("m", "u", "c", emit=emitter), responses)
    emitter.finish(result.content)
    assert drain(emitter)[1] == result.content and "汇总到一半" not in result.content


# ── 出口 ──────────────────────────────────────────────────────────────────────

def test_finish_guarantees_the_visible_text_equals_the_stored_reply():
    for streamed, final, resets in ((["  你好", "\n"], "你好", 0), (["错误的前半句"], "正确回复", 1), ([], "固定话术", 0)):
        emitter = Emitter()
        for piece in streamed:
            emitter.delta(piece)
        emitter.finish(final)
        events, text = drain(emitter)
        assert text.strip() == final and [e for e, _ in events].count("reset") == resets


def test_first_token_time_only_counts_text_that_stays():
    emitter = Emitter()
    emitter.delta("过场话")
    emitter.reset()
    assert emitter.first_ms is None
    emitter.delta("正式回复")
    first = emitter.first_ms
    emitter.delta("继续")
    assert first is not None and emitter.first_ms == first


@pytest.mark.asyncio
async def test_idle_connection_gets_a_ping_and_bad_characters_do_not_break_the_stream(monkeypatch):
    monkeypatch.setattr(streaming, "PING_SECONDS", 0.02)
    emitter = Emitter()
    source = emitter.events()
    assert await anext(source) == b": ping\n\n"
    emitter.delta("坏字符\ud83d")
    assert (await anext(source)).startswith("event: delta\ndata: {\"text\": \"坏字符".encode())
    emitter.close()
    with pytest.raises(StopAsyncIteration):
        await anext(source)


# ── HTTP 接口 ─────────────────────────────────────────────────────────────────

def parse(raw):
    events = []
    for block in raw.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if lines:
            events.append((lines["event"], json.loads(lines["data"])))
    return events


def use_stream(orch, *replies):
    client = streaming_client(*replies)
    for agents in orch._pool.values():
        for agent in agents:
            agent._client = client
    return client


@pytest.mark.asyncio
async def test_stream_endpoint_matches_chat_and_replays_from_the_receipt(services):  # noqa: F811
    memory, orch, _, http = services
    client = use_stream(orch, message_events(("text", "退款", "三到五天", "到账")))
    response = await http.post("/chat/stream", json=body())
    assert response.headers["content-type"].startswith("text/event-stream") and response.headers["x-accel-buffering"] == "no"
    events = parse(response.text)
    done = events[-1][1]
    assert events[-1][0] == "done" and visible(events) == done["response"] == "退款三到五天到账"
    assert [d["stage"] for e, d in events if e == "status"] == ["intent", "answer"]
    assert done["performance"]["first_token_ms"] >= 0 and done["performance"]["llm"]["output_tokens"] == 33
    assert await memory._redis.llen(memory._wm_key("u", "c")) == 2
    again = parse((await http.post("/chat/stream", json=body())).text)
    assert visible(again) == done["response"] and again[-1][1] == done
    assert (await http.post("/chat", json=body())).json() == done and len(client.calls) == 1


@pytest.mark.asyncio
async def test_fixed_replies_are_sent_as_text_too(services):  # noqa: F811
    _, orch, _, http = services
    orch._pool[AgentType.ESCALATION] = [EscalationAgent(None, "m")]
    orch.recognize_intent = AsyncMock(return_value=IntentResult(
        IntentCategory.PAYMENT_ISSUE, .95, UrgencyLevel.CRITICAL, "billing", {}, "test", 0))
    events = parse((await http.post("/chat/stream", json=body("账户正在被盗刷"))).text)
    assert visible(events) == events[-1][1]["response"] and visible(events).startswith("请先确保安全")


@pytest.mark.asyncio
async def test_stream_endpoint_reports_failures_in_band(services):  # noqa: F811
    memory, orch, _, http = services
    use_stream(orch, RuntimeError("down"), RuntimeError("down"), message_events(("text", "恢复")))
    failed = parse((await http.post("/chat/stream", json=body())).text)
    assert failed[-1][0] == "done" and failed[-1][1]["status"] == "failed" and failed[-1][1]["retryable"]
    assert await memory._redis.llen(memory._wm_key("u", "c")) == 0
    conflict = parse((await http.post("/chat/stream", json=body("另一条消息"))).text)
    assert conflict == [("error", {"status": 409, "detail": "同一 client_request_id 不能用于不同消息"})]
    recovered = parse((await http.post("/chat/stream", json=body())).text)
    assert recovered[-1][1]["status"] == "success" and visible(recovered) == "恢复"


@pytest.mark.asyncio
async def test_unexpected_error_does_not_leak_details(services):  # noqa: F811
    _, orch, _, http = services

    async def boom(*args, **kwargs):
        raise RuntimeError("secret-detail")
    orch.run = boom
    raw = (await http.post("/chat/stream", json=body())).text
    assert parse(raw)[-1] == ("error", {"status": 500, "detail": "服务内部错误"}) and "secret-detail" not in raw


@pytest.mark.asyncio
async def test_turn_finishes_after_the_client_disconnects(services):  # noqa: F811
    memory, orch, _, http = services
    release = asyncio.Event()

    class Slow(FakeStream):
        async def __aiter__(self):
            events = message_events(("text", "前半", "后半"))
            for event in events[:3]:
                yield event
            await release.wait()
            for event in events[3:]:
                yield event
    use_stream(orch, Slow([]))
    response = await api.chat_stream(api.ChatRequest(**body()), SimpleNamespace(state=SimpleNamespace(account=None)))
    source = response.body_iterator
    while b"delta" not in await anext(source):
        pass
    await source.aclose()               # 客户端断开
    task = next(iter(api._stream_turns))
    release.set()
    await asyncio.wait_for(task, 5)
    receipt = await memory.get_turn_receipt("u", "c", "one")
    assert receipt["response"]["response"] == "前半后半" and not api._stream_turns


@pytest.mark.asyncio
async def test_stream_endpoint_uses_the_same_access_rules(services, monkeypatch):  # noqa: F811
    _, orch, _, http = services
    use_stream(orch, message_events(("text", "好的")))
    token = "reader-token-for-streaming-tests"
    monkeypatch.setenv("KORA_API_TOKENS", json.dumps({token: {"user_id": "real-user", "role": "user"}}))
    assert (await http.post("/chat/stream", json=body())).status_code == 401
    events = parse((await http.post("/chat/stream", json=body(), headers={"Authorization": f"Bearer {token}"})).text)
    assert events[-1][0] == "done"
    assert await services[0].get_turn_receipt("real-user", "c", "one")      # 身份以令牌为准
