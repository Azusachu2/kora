"""一轮对话的流式出口（SSE）。

事件：
  status  当前阶段，供界面提示；不是回复内容
  delta   追加一段回复正文
  reset   丢弃此前的 delta（转去查资料、换 Agent 重答时发出）
  done    完整的 ChatResponse，正文以它为准
  error   本轮失败
"""
import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, Optional

PING_SECONDS = 10


class Emitter:
    """收集事件；用户最终看到的正文必须与入库的回复一致，由 finish 保证。"""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._started = time.monotonic()
        self._stage: Optional[str] = None
        self.text = ""
        self.first_ms: Optional[float] = None       # 保留到最后的正文的首字耗时

    def send(self, event: str, data: Dict[str, Any]) -> None:
        self._queue.put_nowait((event, data))

    def status(self, stage: str) -> None:
        if stage != self._stage:
            self._stage = stage
            self.send("status", {"stage": stage})

    def delta(self, text: str) -> None:
        if not text:
            return
        if self.first_ms is None:
            self.first_ms = round((time.monotonic() - self._started) * 1000, 1)
        self.text += text
        self.send("delta", {"text": text})

    def reset(self) -> None:
        """已发出的文字作废；其首字耗时也不再算数。"""
        if self.text:
            self.text, self.first_ms = "", None
            self.send("reset", {})

    def finish(self, final: str) -> None:
        if self.text.strip() != final.strip():
            self.reset()
            self.delta(final)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[bytes]:
        """SSE 字节流；长时间没有内容时发注释行，避免代理断开连接。"""
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), PING_SECONDS)
            except TimeoutError:
                yield b": ping\n\n"
                continue
            if item is None:
                return
            event, data = item
            body = json.dumps(data, ensure_ascii=False)
            yield f"event: {event}\ndata: {body}\n\n".encode("utf-8", errors="replace")
