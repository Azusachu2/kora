import asyncio
import json

import httpx
import pytest
from anthropic import AsyncAnthropic

from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


def make_recognizer(threshold=0.5):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True
    recognizer.threshold = threshold
    return recognizer


def vote(recognizer, llm, emb, pat):
    return recognizer._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        {"intent": pat[0], "confidence": pat[1]},
    )


def test_pattern_cannot_refine_a_different_intent_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.BILLING, 0.9),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.BILLING
    assert confidence == pytest.approx(0.63)
    assert "refined_by_pattern" not in sources


def test_coarse_and_fine_scores_choose_the_same_group_together():
    intent, confidence, sources = vote(
        make_recognizer(threshold=0.2),
        (IntentCategory.BILLING, 0.3),       # weighted score 0.21
        (IntentCategory.TECHNICAL_LOGIN, 0.9),  # weighted score 0.18
        (IntentCategory.TECHNICAL, 0.4),     # weighted score 0.04
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.22)
    assert "refined_by_pattern" not in sources


def test_pattern_can_still_refine_within_the_winning_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.TECHNICAL, 0.8),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.61)
    assert sources["refined_by_pattern"] == pytest.approx(0.5)


# ── 置信度门槛与降级 ──────────────────────────────────────────────────────────

def test_confident_llm_is_not_downgraded_when_other_branches_miss():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.BILLING, 0.6),          # 融合分 0.42
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.OTHER, 0.0),
    )
    assert intent is IntentCategory.BILLING
    assert confidence == pytest.approx(0.42)
    assert sources["accepted_by_llm"] == pytest.approx(0.6)


def test_unsure_llm_is_still_downgraded_to_other():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.BILLING, 0.45),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.OTHER, 0.0),
    )
    assert intent is IntentCategory.OTHER and confidence < 0.5
    assert "accepted_by_llm" not in sources


def test_failed_llm_does_not_turn_weak_guesses_into_intents():
    recognizer = make_recognizer()
    failed = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
    weak = {"intent": IntentCategory.INVOICE, "confidence": 0.2}
    strong = {"intent": IntentCategory.INVOICE, "confidence": 0.8}
    generic = {"intent": IntentCategory.QUERY, "confidence": 0.5}
    specific = {"intent": IntentCategory.LOGISTICS, "confidence": 0.5}
    escalation = {"intent": IntentCategory.ESCALATION, "confidence": 0.5}
    assert recognizer._vote(failed, weak, escalation)[:2] == (IntentCategory.ESCALATION, 0.5)
    assert recognizer._vote(failed, weak, generic)[:2] == (IntentCategory.OTHER, 0.0)
    assert recognizer._vote(failed, strong, generic)[:2] == (IntentCategory.INVOICE, 0.8)
    assert recognizer._vote(failed, weak, specific)[:2] == (IntentCategory.LOGISTICS, 0.5)


def run_with(body, monkeypatch, calls, message="我想了解一下", model="claude-opus-5-5"):
    """在 HTTP 边界替换模型响应，检查真实请求体。body 为 None 时返回 529。"""
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")

    async def run():
        def handle(request):
            calls.append(json.loads(request.content))
            if body is None:
                return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "x"}})
            return httpx.Response(200, json={
                "id": "m", "type": "message", "role": "assistant", "model": model, "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]})
        recognizer = IntentRecognizer("test", model=model)
        recognizer.client = AsyncAnthropic(api_key="test", base_url="https://api.anthropic.com", max_retries=0,
                                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        first = await recognizer.recognize(message)
        second = await recognizer.recognize(message)
        await recognizer.client.close()
        return first, second
    return asyncio.run(run())


@pytest.mark.parametrize("model,effort", [("claude-opus-5-5", {"effort": "low"}), ("claude-haiku-4-5", None), ("deepseek-chat", None)])
def test_intent_request_leaves_room_for_thinking(monkeypatch, model, effort):
    calls = []
    run_with({"intent": "query", "confidence": 0.9, "urgency": "low"}, monkeypatch, calls, model=model)
    assert calls[0]["max_tokens"] == 1024 and calls[0].get("output_config") == effort
    calls.clear()
    monkeypatch.setenv("KORA_INTENT_MAX_TOKENS", "2048")
    monkeypatch.setenv("KORA_INTENT_EFFORT", "")
    run_with({"intent": "query", "confidence": 0.9}, monkeypatch, calls, model=model)
    assert calls[0]["max_tokens"] == 2048 and "output_config" not in calls[0]


def test_failed_result_is_not_cached_but_success_is(monkeypatch):
    calls = []
    run_with(None, monkeypatch, calls)
    assert len(calls) == 2
    calls.clear()
    run_with({"intent": "query", "confidence": 0.9, "urgency": "low"}, monkeypatch, calls)
    assert len(calls) == 1


# ── 紧急度 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("message,intent,urgency,expected", [
    ("有人正用我的余额下单", "account_security", "critical", UrgencyLevel.CRITICAL),
    ("麻烦立刻发我一份价目表", "query", "low", UrgencyLevel.LOW),
    ("紧急出口标识怎么贴", "query", "low", UrgencyLevel.LOW),
    ("帮我接客服", "human_handoff", "low", UrgencyLevel.HIGH),
    ("麻烦立刻发我一份价目表", "query", None, UrgencyLevel.CRITICAL),       # 缺失时回到关键词
    ("麻烦立刻发我一份价目表", "query", "unknown", UrgencyLevel.CRITICAL),
])
def test_llm_urgency_overrides_keywords(monkeypatch, message, intent, urgency, expected):
    body = {"intent": intent, "confidence": 0.9}
    if urgency is not None:
        body["urgency"] = urgency
    result, _ = run_with(body, monkeypatch, [], message=message)
    assert result.intent.value == intent and result.urgency is expected


def test_keywords_still_cover_urgency_when_llm_fails(monkeypatch):
    result, _ = run_with(None, monkeypatch, [], message="紧急情况需要处理")
    assert result.urgency is UrgencyLevel.CRITICAL


# ── 数字与关键词误触 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("message,codes", [
    ("打开就报404", ["404"]), ("接口返回502，重试也一样", ["502"]), ("错误码429是什么意思", ["429"]),
    ("HTTP 503", ["503"]), ("登录页提示 403。", ["403"]), ("系统提示401和500交替出现", ["401", "500"]),
    ("我付了450元", []), ("买了500g咖啡豆", []), ("报销 500 的餐费", []), ("返回500积分到账户", []),
    ("订单号 502 的页面打不开", []), ("共 480 人报名", []), ("系统里显示 4500 条", []), ("版本 5.500 登录不了", []),
    ("住在 503 室，页面地址填错了", []), ("第 404 次刷新", []), ("充 500 提示失败", []),
])
def test_error_codes_need_fault_context_and_are_not_quantities(message, codes):
    assert make_recognizer()._extract_entities(message)["error_code"] == codes


@pytest.mark.parametrize("message,order_ids", [
    ("my order never arrived", []), ("订单编号 K20269981", ["K20269981"]), ("订单号：77120034", ["77120034"]), ("order #A-1029", ["A-1029"]),
])
def test_order_id_requires_a_digit(message, order_ids):
    assert make_recognizer()._extract_entities(message)["order_id"] == order_ids


@pytest.mark.parametrize("message,intent", [
    ("this is which version", IntentCategory.OTHER), ("hi there", IntentCategory.GREETING),
    ("my refunds", IntentCategory.REFUND), ("花了450元买的", IntentCategory.OTHER),
    ("接口报502", IntentCategory.TECHNICAL_CRASH), ("登录提示403", IntentCategory.TECHNICAL_LOGIN),
])
def test_pattern_keywords_respect_boundaries(message, intent):
    assert make_recognizer()._pattern_recognize(message)["intent"] is intent


@pytest.mark.parametrize("message,intent,agents", [
    ("退我450元", IntentCategory.REFUND, [AgentType.BILLING]),
    ("订单 5008812 的发票还没开", IntentCategory.INVOICE, [AgentType.BILLING]),
    ("支付接口报502", IntentCategory.PAYMENT_ISSUE, [AgentType.BILLING, AgentType.TECHNICAL]),
])
def test_numbers_only_bring_in_technical_agent_as_error_codes(message, intent, agents):
    entities = make_recognizer()._extract_entities(message)
    decision = AgentOrchestrator(api_key="test")._route_decision(Request(
        message=message, user_id="u", conv_id="c", intent=intent, urgency=UrgencyLevel.LOW, entities=entities))
    assert decision.agent_types == agents


@pytest.mark.parametrize("urgency,agent,escalated", [
    (UrgencyLevel.CRITICAL, AgentType.ESCALATION, True), (UrgencyLevel.LOW, AgentType.GENERAL, False)])
def test_critical_message_is_escalated_instead_of_clarified(urgency, agent, escalated):
    result = asyncio.run(AgentOrchestrator(api_key="test").run(Request(
        message="家里的设备着火了", user_id="u", conv_id="c", intent=IntentCategory.OTHER, urgency=urgency, intent_confidence=0.42)))
    assert result.agent_type is agent and result.escalated is escalated


# ── 紧急升级的安全提示 ────────────────────────────────────────────────────────

def escalation_reply(intent, urgency):
    return asyncio.run(AgentOrchestrator(api_key="test").run(Request(
        message="x", user_id="u", conv_id="c", intent=intent, urgency=urgency, intent_confidence=0.9))).response


@pytest.mark.parametrize("intent,extra", [
    (IntentCategory.COMPLAINT, None), (IntentCategory.OTHER, None), (None, None),
    (IntentCategory.ACCOUNT_SECURITY, "冻结"), (IntentCategory.PAYMENT_ISSUE, "冻结"),
    (IntentCategory.TECHNICAL_CRASH, "覆盖数据"),
])
def test_critical_reply_starts_with_safety_notice(intent, extra):
    request = Request(message="x", user_id="u", conv_id="c", intent=intent, urgency=UrgencyLevel.CRITICAL)
    text = asyncio.run(AgentOrchestrator(api_key="test")._pool[AgentType.ESCALATION][0].handle(request)).content
    # 人身安全提示与意图无关，始终排在最前
    assert text.startswith("请先确保安全") and "120" in text and "119" in text
    assert text.index("120") < text.index("情况紧急，需要人工客服尽快处理")
    for word in ("冻结", "覆盖数据"):
        assert (word in text) is (word == extra)


@pytest.mark.parametrize("intent,urgency,opening", [
    (IntentCategory.HUMAN_HANDOFF, UrgencyLevel.HIGH, "好的，这个问题转给人工客服处理。"),
    (IntentCategory.ESCALATION, UrgencyLevel.HIGH, "这个问题需要人工客服跟进处理。")])
def test_ordinary_handoff_has_no_safety_notice(intent, urgency, opening):
    text = escalation_reply(intent, urgency)
    assert text.startswith(opening) and "120" not in text
