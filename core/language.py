"""判断用户使用中文还是英文，并提供两种语言的固定话术。"""
import re
from typing import Any, Dict, List, Optional

_CJK = re.compile(r"[㐀-鿿]")
_WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
DEFAULT = "zh"


def _decide(text: str) -> Optional[str]:
    """看不出来时返回 None。中文里夹英文型号、错误码很常见，所以英文词要明显更多才算英文。"""
    cjk, words = len(_CJK.findall(text)), [w for w in _WORD.findall(text) if len(w) > 1]
    if not cjk and not words:
        return None
    return "en" if len(words) > 2 * cjk else "zh"


def detect(message: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """ok、thanks、订单号这类短消息看不出语言，沿用用户上一句的语言。"""
    text = message or ""
    language = _decide(text)
    short = not _CJK.search(text) and len(_WORD.findall(text)) <= 2
    if language is None or (language == "en" and short):
        for item in reversed(history or []):
            if item.get("role") == "user" and (previous := _decide(str(item.get("content", "")))):
                return previous
    return language or DEFAULT


TEXTS: Dict[str, Dict[str, str]] = {
    "clarify": {
        "zh": "我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？",
        "en": "I'm not sure yet what you need help with. Is it about an order or delivery, a refund or bill, "
              "your account details, or a technical problem?",
    },
    "agent_failed": {
        "zh": "抱歉，处理您的请求时出现问题，请稍后重试。",
        "en": "Sorry, something went wrong while handling your request. Please try again later.",
    },
    "agent_unavailable": {
        "zh": "服务暂时不可用，请稍后重试。",
        "en": "The service is temporarily unavailable. Please try again later.",
    },
    "all_failed": {
        "zh": "抱歉，所有 Agent 均处理失败。",
        "en": "Sorry, we couldn't process your request. Please try again later.",
    },
    "supplement": {"zh": "补充说明：", "en": "Additional information:"},
    "outcome_unknown": {
        "zh": "该请求涉及可能修改数据的工具，请先核验执行结果再发起新的请求。",
        "en": "This request may already have changed data. Please check the result before sending a new request.",
    },
    "safety_intro": {"zh": "请先确保安全：", "en": "Please make sure you are safe first:"},
    "safety_person": {
        "zh": "如果有人受伤、身体不适，或有起火、冒烟等危险，请先远离危险，"
              "并拨打当地急救电话（中国大陆：急救 120、火警 119、报警 110）。",
        "en": "If anyone is hurt or unwell, or there is fire or smoke, move away from the danger and call your local "
              "emergency number (mainland China: ambulance 120, fire 119, police 110).",
    },
    "safety_funds": {
        "zh": "如果资金或账户正在被盗用，请马上联系银行或支付平台冻结，"
              "修改登录和支付密码，保留交易记录，必要时报警。",
        "en": "If your money or account is being misused, contact your bank or payment provider now to freeze it, "
              "change your login and payment passwords, keep the transaction records, and report it to the police if needed.",
    },
    "safety_outage": {
        "zh": "如果是服务故障或数据丢失，请先停止可能覆盖数据的操作，记下开始时间、影响范围和错误提示。",
        "en": "If this is an outage or data loss, stop anything that could overwrite data, and note when it started, "
              "what is affected and any error messages.",
    },
    # 转人工交接：按情况开头，只列有值的信息，不出现意图代码、紧急度等内部字段。
    "handoff_request": {"zh": "好的，这个问题转给人工客服处理。", "en": "Sure, I'll pass this to a human agent."},
    "handoff_followup": {"zh": "这个问题需要人工客服跟进处理。", "en": "This needs follow-up from a human agent."},
    "handoff_urgent": {"zh": "情况紧急，需要人工客服尽快处理。", "en": "This is urgent and needs a human agent as soon as possible."},
    "handoff_details": {"zh": "联系人工客服时提供这些信息，可以处理得更快：", "en": "Share these details with the agent to speed things up:"},
    "handoff_channel": {"zh": "目前这里还不能直接接通人工客服，请通过官方客服渠道联系。",
                        "en": "I can't connect you to a human agent from here yet, so please contact support through the official channels."},
    "handoff_contact": {"zh": "人工客服联系方式：{contact}", "en": "How to reach a human agent: {contact}"},
    "handoff_privacy": {"zh": "请不要发送密码、短信验证码或完整支付凭证。", "en": "Please don't share passwords, SMS codes or full payment details."},
    "field_topic": {"zh": "问题类型", "en": "Topic"},
    "field_order_id": {"zh": "订单号", "en": "Order ID"},
    "field_amount": {"zh": "金额", "en": "Amount"},
    "field_date": {"zh": "日期", "en": "Date"},
    "field_error_code": {"zh": "错误码", "en": "Error code"},
    "field_product": {"zh": "商品", "en": "Product"},
    "topic_commerce": {"zh": "订单、支付与售后", "en": "orders, payments and after-sales"},
    "topic_technical": {"zh": "技术问题", "en": "a technical issue"},
    "topic_account": {"zh": "账户问题", "en": "an account issue"},
    # 以下写给模型
    "reply_language": {
        "zh": "",
        "en": "\n\n[Reply language]\nThe user writes in English. Reply in English only, including headings and quoted "
              "policies (translate Chinese sources). Aim for about 120–220 English words instead of the Chinese "
              "character guidance above.",
    },
    "composer_language": {
        "zh": "只输出给用户看的中文回复，不要提及 Agent。\n\n"
              "默认正文约 300–500 个中文字符，分清各问题的结论和下一步，不重复展开背景说明；用户要求详细时再展开。\n",
        "en": "只输出给用户看的英文回复（English only），不要提及 Agent。\n\n"
              "默认正文约 180–300 个英文单词，分清各问题的结论和下一步，不重复展开背景说明；用户要求详细时再展开。\n",
    },
}


def text(key: str, language: str, **values: Any) -> str:
    template = TEXTS[key].get(language) or TEXTS[key][DEFAULT]
    return template.format(**values) if values else template
