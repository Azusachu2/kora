"""LLM response helpers shared by Anthropic-compatible providers."""
import json
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List
from urllib.parse import urlsplit
import time

from core.performance import record_llm_call

# 这些模型默认开启思考，思考与正文共用 max_tokens；短输出任务用 effort 控制思考量。
# 它们同时不接受自定义 temperature。
_EFFORT_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5", "claude-fable-5")


def fast_model(default: str) -> str:
    """意图、查询改写、记忆摘要等辅助调用使用的模型；未配置时与主模型相同。"""
    return os.getenv("KORA_FAST_MODEL", "").strip() or default


def _request(client: Any, effort: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt one request to the model and provider; the caller's dict is not modified.

    DeepSeek enables thinking by default, and those tokens share max_tokens with
    the answer. A short JSON request can therefore finish with thinking only.
    Use extra_body for compatibility with the pinned Anthropic SDK; leave other
    providers and explicitly configured thinking modes unchanged.
    """
    kwargs = dict(kwargs)
    if str(kwargs.get("model", "")).startswith(_EFFORT_MODELS):
        kwargs.pop("temperature", None)
        if effort:
            # 固定版本的 SDK 没有 output_config 参数，经 extra_body 传递。
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("output_config", {"effort": effort})
            kwargs["extra_body"] = extra_body
    if urlsplit(str(getattr(client, "base_url", ""))).hostname == "api.deepseek.com":
        extra_body = dict(kwargs.get("extra_body") or {})
        if "thinking" not in kwargs and "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = extra_body
    return kwargs


async def create_message(client: Any, *, stage: str = "unspecified", effort: str = "", **kwargs: Any) -> Any:
    """一次完整调用；模型与供应商差异见 _request。"""
    kwargs = _request(client, effort, kwargs)
    started, response, status = time.monotonic(), None, "error"
    try:
        response = await client.messages.create(**kwargs)
        status = "success"
        return response
    finally:
        record_llm_call(stage, kwargs.get("model", "unknown"), started, status, response)


class StreamInterrupted(RuntimeError):
    """连接在回复结束前断开；已收到的内容不完整，不能当作回复。"""


async def stream_message(client: Any, *, on_text: Callable[[str], None], stage: str = "unspecified",
                         effort: str = "", **kwargs: Any) -> Any:
    """流式调用：正文逐段交给 on_text，返回与 create_message 等价的消息。

    自行累积内容块。固定版本的 SDK 累积器会丢掉思考块的签名，而签名必须原样带回工具循环。
    """
    kwargs = _request(client, effort, kwargs)
    message = SimpleNamespace(content=[], stop_reason=None, usage={}, model=None)
    blocks: Dict[int, Dict[str, Any]] = {}
    tool_json: Dict[int, str] = {}
    texts, finished = 0, False
    started, status = time.monotonic(), "error"
    try:
        stream = await client.messages.create(stream=True, **kwargs)
        try:
            async for raw in stream:
                event = _plain(raw)
                kind, index = event.get("type"), event.get("index")
                if kind == "message_start":
                    message.model = event["message"].get("model")
                    message.usage.update(event["message"].get("usage") or {})
                elif kind == "content_block_start":
                    block = blocks[index] = dict(event["content_block"])
                    if block.get("type") == "text":
                        # 与 extract_text_content 一致：多段正文之间换行。
                        texts += 1
                        if texts > 1:
                            on_text("\n")
                        on_text(block.get("text") or "")
                elif kind == "content_block_delta":
                    _apply_delta(blocks[index], event["delta"], tool_json, index, on_text)
                elif kind == "message_delta":
                    message.stop_reason = event["delta"].get("stop_reason")
                    message.usage.update(event.get("usage") or {})
                elif kind == "message_stop":
                    finished = True
        finally:
            await stream.close()
        if not finished:
            raise StreamInterrupted("模型流在结束前中断")
        for index, text in tool_json.items():
            if text.strip():
                blocks[index]["input"] = json.loads(text)
        # 空正文块带回工具循环会被接口拒绝。
        message.content = [blocks[i] for i in sorted(blocks)
                           if blocks[i].get("type") != "text" or blocks[i].get("text")]
        status = "success"
        return message
    finally:
        record_llm_call(stage, kwargs.get("model", "unknown"), started, status, message)


def _apply_delta(block: Dict[str, Any], delta: Dict[str, Any], tool_json: Dict[int, str], index: int,
                 on_text: Callable[[str], None]) -> None:
    kind = delta.get("type")
    if kind == "text_delta":
        block["text"] = block.get("text", "") + delta["text"]
        on_text(delta["text"])
    elif kind == "input_json_delta":
        tool_json[index] = tool_json.get(index, "") + delta["partial_json"]
    elif kind == "thinking_delta":
        block["thinking"] = block.get("thinking", "") + delta["thinking"]
    elif kind == "signature_delta":
        block["signature"] = block.get("signature", "") + delta["signature"]


def _plain(value: Any) -> Any:
    """SDK 事件转普通字典。固定版本不认识思考块，其 to_dict 会抛错，因此按字段读取。"""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "__pydantic_fields_set__"):
        fields = {**value.__dict__, **(value.__pydantic_extra__ or {})}
        return {key: _plain(item) for key, item in fields.items() if item is not None}
    return value


def extract_text_content(content: Iterable[Any], separator: str = "\n") -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return separator.join(t for t in texts if t)
