"""转人工：给用户看的交接说明不含内部代码；回复里提到人工客服不算升级。"""
import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.agent_orchestrator import AgentOrchestrator, AgentType, EscalationAgent, GeneralAgent, Request, ResponseComposer
from core.intent_recognizer import IntentCategory, UrgencyLevel

INTERNAL = ("human_handoff", "escalation", "HIGH", "CRITICAL", "LOW", "意图=", "紧急度", "intent=", "urgency", "[]", "{")


def handoff(message, intent=IntentCategory.HUMAN_HANDOFF, urgency=UrgencyLevel.HIGH, **kwargs):
    req = Request(message, "u", "c", intent=intent, urgency=urgency, **kwargs)
    return asyncio.run(EscalationAgent(SimpleNamespace(), "m").handle(req))


@pytest.mark.parametrize("message", ["我需要升级人工", "I want to talk to a human agent"])
def test_handoff_reply_has_no_internal_codes(message):
    response = handoff(message, entities={"order_id": [], "amount": []})
    assert response.escalate and not any(code in response.content for code in INTERNAL)
    assert ("人工客服" in response.content) or ("human agent" in response.content)


def test_handoff_lists_only_known_details_with_readable_labels():
    text = handoff("转人工", entities={"order_id": ["A12345"], "amount": ["500元", "20元"], "date": []},
                   conversation_state={"domain": "commerce"}).content
    assert "- 问题类型：订单、支付与售后" in text and "- 订单号：A12345" in text and "- 金额：500元、20元" in text
    assert "日期" not in text and "联系人工客服时提供这些信息" in text
    english = handoff("I need a human", entities={"order_id": ["A12345"]}, conversation_state={"domain": "technical"}).content
    assert "- Topic: a technical issue" in english and "- Order ID: A12345" in english


@pytest.mark.parametrize("domain", ["general", "commerce"])
def test_without_details_there_is_no_empty_list(domain):
    text = handoff("转人工", conversation_state={"domain": domain}).content      # “你是人机吗”也会落到 commerce
    assert "提供这些信息" not in text and "问题类型" not in text and text.startswith("好的，这个问题转给人工客服处理。")


def test_contact_comes_from_configuration(monkeypatch):
    assert "联系方式" not in handoff("转人工").content
    monkeypatch.setenv("KORA_HANDOFF_CONTACT", "在线客服每天 9:00–21:00")
    assert "人工客服联系方式：在线客服每天 9:00–21:00" in handoff("转人工").content


def test_mentioning_human_support_is_not_an_escalation():
    reply = "我是 Kora 智能客服，不是真人。需要人工客服的话，可以直接说“转人工”。"
    create = AsyncMock(return_value=SimpleNamespace(content=[{"type": "text", "text": reply}], stop_reason="end_turn"))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {AgentType.GENERAL: [GeneralAgent(client, "m")], AgentType.ESCALATION: [EscalationAgent(client, "m")]}
    orch._composer = ResponseComposer(client, "m")
    orch._recent_tool_traces = deque(maxlen=5)
    result = asyncio.run(orch.run(Request("你是人机吗", "u", "c", intent=IntentCategory.QUERY, urgency=UrgencyLevel.LOW, intent_confidence=0.9)))
    assert result.response == reply and result.escalated is False
    handed = asyncio.run(orch.run(Request("我需要升级人工", "u", "c", intent=IntentCategory.HUMAN_HANDOFF,
                                          urgency=UrgencyLevel.HIGH, intent_confidence=0.9)))
    assert handed.escalated is True and handed.agent_type is AgentType.ESCALATION


def test_agents_are_told_not_to_promise_a_transfer():
    prompt = GeneralAgent(SimpleNamespace(), "m")._build_system_prompt(Request("你是人机吗", "u", "c"))
    assert "不能直接接通人工客服" in prompt and "不要说你已经或将会替他接通人工" in prompt
    assert "talk to a human" in prompt          # 英文用户不应被要求输入中文“转人工”
