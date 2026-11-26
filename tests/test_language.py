"""中英文识别与固定话术。"""
import asyncio
import pathlib
import re
import string
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, BillingAgent, EscalationAgent, GeneralAgent, Request, ResponseComposer,
)
from core import language
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.skill_loader import SkillManager

ROOT = pathlib.Path(__file__).resolve().parent.parent
CJK = re.compile(r"[㐀-鿿]")
ZH = [{"role": "user", "content": "我的订单到哪了"}, {"role": "assistant", "content": "Please share the order ID"}]
EN = [{"role": "user", "content": "Where is my order?"}, {"role": "assistant", "content": "请提供订单号"}]


@pytest.mark.parametrize("message,history,expected", [
    ("退款多久到账", None, "zh"),
    ("Where is my refund?", None, "en"),
    ("我的 iPhone 15 Pro Max 充不进电", None, "zh"),          # 中文夹英文型号
    ("退款报 error code E1001", None, "zh"),
    ("Where is my 小米 phone, it was due yesterday", None, "en"),
    ("ok", ZH, "zh"), ("thank you", ZH, "zh"), ("ok", EN, "en"), ("ok", None, "en"),
    ("Where is my parcel?", ZH, "en"),                       # 明确换成英文
    ("退款", EN, "zh"),
    ("20260101-889", EN, "en"), ("20260101-889", ZH, "zh"),  # 订单号沿用上一句
    ("", None, "zh"), ("？", None, "zh"), ("😀", EN, "en"),
])
def test_detect(message, history, expected):
    assert language.detect(message, history) == expected
    assert Request(message, "u", "c", history=history).language == expected


def test_every_text_exists_in_both_languages_with_the_same_placeholders():
    fields = lambda s: {name for _, name, _, _ in string.Formatter().parse(s) if name}
    for key, versions in language.TEXTS.items():
        assert set(versions) == {"zh", "en"} and fields(versions["zh"]) == fields(versions["en"]), key
    user_facing = set(language.TEXTS) - {"composer_language"}
    assert not [key for key in user_facing if CJK.search(language.TEXTS[key]["en"])]


def critical(message, intent):
    req = Request(message, "u", "c", intent=intent, urgency=UrgencyLevel.CRITICAL, entities={"order_id": ["A1"]})
    return asyncio.run(EscalationAgent(SimpleNamespace(), "m").handle(req)).content


def test_escalation_and_safety_tips_follow_the_user_language():
    text = critical("Someone is making payments from my account right now", IntentCategory.PAYMENT_ISSUE)
    assert not CJK.search(text) and text.startswith("Please make sure you are safe first")
    assert "120" in text and "freeze" in text and "passwords" in text and "- Order ID: A1" in text
    assert "outage" in critical("The whole site is down and orders are lost", IntentCategory.TECHNICAL)
    chinese = critical("账户正在被盗刷", IntentCategory.PAYMENT_ISSUE)
    assert chinese.startswith("请先确保安全") and "冻结" in chinese


def test_clarification_and_failures_follow_the_user_language():
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._recent_tool_traces, orch._pool = deque(), {}
    req = Request("can you do the thing for me", "u", "c", intent=IntentCategory.OTHER, intent_confidence=0.2)
    assert asyncio.run(orch.run(req)).response == language.TEXTS["clarify"]["en"]
    assert asyncio.run(orch._execute(req, AgentType.GENERAL)).content == language.TEXTS["agent_unavailable"]["en"]
    down = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("down"))))
    assert asyncio.run(GeneralAgent(down, "m").handle(req)).content == language.TEXTS["agent_failed"]["en"]
    failed = [AgentResponse(AgentType.BILLING, "", False)]
    assert asyncio.run(ResponseComposer(down, "m").compose(req, failed)) == language.TEXTS["all_failed"]["en"]
    both = [AgentResponse(kind, kind.value, True) for kind in (AgentType.BILLING, AgentType.TECHNICAL)]
    assert "Additional information:" in asyncio.run(ResponseComposer(down, "m").compose(req, both))


def test_models_are_told_which_language_to_use():
    skills = SkillManager(root_dir=str(ROOT / "skills"))
    skills.load()
    agent = BillingAgent(SimpleNamespace(), "m", skills)
    english = agent._build_system_prompt(Request("How do I get a refund?", "u", "c"))
    assert "[动态 Skills]" in english and english.rstrip().endswith(language.TEXTS["reply_language"]["en"].strip())
    assert "Reply language" not in agent._build_system_prompt(Request("怎么申请退款", "u", "c"))
    create = AsyncMock(return_value=SimpleNamespace(content=[{"type": "text", "text": "ok"}], stop_reason="end_turn"))
    composer = ResponseComposer(SimpleNamespace(messages=SimpleNamespace(create=create)), "m")
    both = [AgentResponse(kind, kind.value, True) for kind in (AgentType.BILLING, AgentType.TECHNICAL)]
    for message, expected in (("I was charged twice and cannot log in", "English only"), ("登录失败还被重复扣款", "中文回复")):
        asyncio.run(composer.compose(Request(message, "u", "c"), both))
        assert expected in create.call_args.kwargs["messages"][0]["content"]


@pytest.mark.parametrize("message,agent", [
    ("I was charged twice for my subscription", "billing"), ("Where is my order?", "general"), ("I can't log in", "technical"),
])
def test_business_rules_apply_to_english_messages(message, agent):
    skills = SkillManager(root_dir=str(ROOT / "skills"))
    skills.load()
    assert skills.prompt_for(message, agent)
