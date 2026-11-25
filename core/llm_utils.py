"""LLM response helpers shared by Anthropic-compatible providers."""
from typing import Any, Iterable, List
from urllib.parse import urlsplit
import time

from core.performance import record_llm_call

# 这些模型默认开启思考，思考与正文共用 max_tokens；短输出任务用 effort 控制思考量。
_EFFORT_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5", "claude-fable-5")


async def create_message(client: Any, *, stage: str = "unspecified", effort: str = "", **kwargs: Any) -> Any:
    """Call bounded support tasks without DeepSeek's implicit thinking budget.

    DeepSeek enables thinking by default, and those tokens share max_tokens with
    the answer. A short JSON request can therefore finish with thinking only.
    Use extra_body for compatibility with the pinned Anthropic SDK; leave other
    providers and explicitly configured thinking modes unchanged.
    """
    # Opus 5.5 rejects temperature, including the values used by legacy callers.
    # Keep other models' sampling settings unchanged.
    if kwargs.get("model") == "claude-opus-5-5":
        kwargs.pop("temperature", None)
    if effort and str(kwargs.get("model", "")).startswith(_EFFORT_MODELS):
        # 固定版本的 SDK 没有 output_config 参数，经 extra_body 传递。
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("output_config", {"effort": effort})
        kwargs["extra_body"] = extra_body
    if urlsplit(str(getattr(client, "base_url", ""))).hostname == "api.deepseek.com":
        extra_body = dict(kwargs.get("extra_body") or {})
        if "thinking" not in kwargs and "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = extra_body
    started, response, status = time.monotonic(), None, "error"
    try:
        response = await client.messages.create(**kwargs)
        status = "success"
        return response
    finally:
        record_llm_call(stage, kwargs.get("model", "unknown"), started, status, response)


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
