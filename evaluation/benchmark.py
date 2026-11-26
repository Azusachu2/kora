"""基准评测：意图与路由、知识库检索、端到端耗时。

在项目根目录运行：
  python -m evaluation.benchmark intent                    # 不调用模型：向量 + 关键词降级路径、实体、紧急度、路由
  python -m evaluation.benchmark retrieval                 # 不调用模型：生产 KnowledgeBase 的向量检索
  python -m evaluation.benchmark intent --mode live        # 调用 .env 中的模型，产生费用
  python -m evaluation.benchmark retrieval --lang en       # 英文查询检索同一份中文语料
  python -m evaluation.benchmark retrieval --mode live     # 改写 → 召回 → 重排，产生费用
  python -m evaluation.benchmark e2e --mode live --limit 30 --stream
  常用参数：--split dev|test|all  --reps 3  --out a.json  --compare evaluation/baselines/intent-offline.json

调优只看 dev，最后用 test 报一次数。两者按标签和意图随机分层，划分与得分无关。

intent.jsonl 标注约定：
  - intent 是主意图，有细粒度业务意图时不用大类；accept 是同样说得通的其他意图，只计入宽松准确率。
  - agents 是可接受的 Agent 集合列表；null 表示归属未定，不计路由分（当前为账户类）。
  - critical 只标真实的高风险紧急情况（盗刷、数据泄露、大面积故障、人身安全），不看是否出现"紧急"二字。
  - entities 中 order_id、amount、error_code 始终计分（未写表示应为空）；date 只在写出时计分。
  - 全部用例为合成数据，反映覆盖面而不是线上流量分布。
"""
import argparse
import asyncio
import collections
import contextlib
import contextvars
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent_orchestrator import AgentOrchestrator, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import fast_model
from core.performance import collect_performance, performance_snapshot
from core.streaming import Emitter

DATASETS = ROOT / "evaluation" / "datasets"
ALWAYS_SCORED = ("order_id", "amount", "error_code")
MODES = {"intent": ("offline", "live", "oracle", "majority"),
         "retrieval": ("vector", "lexical", "live", "oracle", "null"),
         "e2e": ("live",)}
# 逐用例可比较的二值指标
BINARY = {"intent": ("ok", "ok_lenient", "entity_ok", "critical_ok", "route_ok", "route_gold_ok"),
          "retrieval": ("hit1", "hit5", "answer5"),
          "e2e": ("success",)}
TOP_K = 5
_observed = contextvars.ContextVar("benchmark_observed", default=None)


# ── 数据 ──────────────────────────────────────────────────────────────────────

def load_jsonl(name):
    return [json.loads(line) for line in (DATASETS / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def fingerprint(*datasets):
    """按内容计算，不受 Git 换行转换影响。"""
    return hashlib.sha256(json.dumps(datasets, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]


def select(rows, split="all", limit=0, group=None):
    """按 split 过滤；limit 时在各组间轮流抽取，保证小样本也覆盖所有类别。"""
    rows = [r for r in rows if split == "all" or r["split"] == split]
    if not limit or limit >= len(rows):
        return rows
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[group(row) if group else ""].append(row)
    picked, depth = [], 0
    while len(picked) < limit:
        for bucket in buckets.values():
            if depth < len(bucket) and len(picked) < limit:
                picked.append(bucket[depth])
        depth += 1
    return picked


# ── 统计 ──────────────────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    center, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def rate(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    k = sum(values)
    return {"rate": round(k / len(values), 4) if values else None, "n": len(values), "ci95": wilson(k, len(values))}


def pct(values, q, digits=1):
    """最近秩百分位。"""
    values = sorted(values)
    return round(values[max(0, math.ceil(q / 100 * len(values)) - 1)], digits) if values else None


def mcnemar(fixed, broken):
    """配对二项精确检验的双侧 p 值。"""
    n = fixed + broken
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broken) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def auc(positive, negative):
    """正例分数高于负例的概率；0.5 表示分数无法区分。"""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return round(wins / (len(positive) * len(negative)), 4)


def cost_summary(rows):
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    calls = sum(r.get("llm_calls", 0) for r in rows)
    return {"latency_ms": {"p50": pct(latencies, 50), "p95": pct(latencies, 95), "max": pct(latencies, 100)},
            "llm_calls": calls, "llm_calls_per_case": round(calls / len(rows), 2) if rows else 0,
            "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("output_tokens", 0) for r in rows)}


# ── 模型调用观测 ──────────────────────────────────────────────────────────────

class SpyClient:
    """透传 SDK 调用，记录实际响应的模型、停止原因和异常类型（不记录内容）。"""

    def __init__(self, client):
        self._client = client
        self.messages = self

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def create(self, **kwargs):
        call = {"requested": kwargs.get("model")}
        observed = _observed.get()
        if observed is not None:
            observed.append(call)
        try:
            response = await self._client.messages.create(**kwargs)
        except asyncio.CancelledError:
            call["error"] = "timeout"
            raise
        except Exception as ex:
            status = getattr(ex, "status_code", None)
            call["error"] = type(ex).__name__ + (f":{status}" if status else "")
            raise
        if kwargs.get("stream"):
            return SpyStream(response, call)
        call["served"] = getattr(response, "model", None)
        call["stop"] = getattr(response, "stop_reason", None)
        return response


class SpyStream:
    """透传流式事件，只记下实际模型、停止原因、内容块类型和中断，不记内容。"""

    def __init__(self, stream, call):
        self._stream, self._call = stream, call

    def __getattr__(self, name):
        return getattr(self._stream, name)

    async def __aiter__(self):
        try:
            async for event in self._stream:
                if event.type == "message_start":
                    self._call["served"] = event.message.model
                elif event.type == "message_delta":
                    self._call["stop"] = event.delta.stop_reason
                elif event.type == "content_block_start":
                    self._call.setdefault("blocks", []).append(event.content_block.type)
                yield event
        except asyncio.CancelledError:
            self._call["error"] = "timeout"
            raise
        except Exception as ex:
            self._call["error"] = type(ex).__name__
            raise


def llm_status(failed, calls):
    """区分基础设施失败与模型输出问题，避免混进准确率里看不出来。"""
    if not failed:
        return "ok"
    last = calls[-1] if calls else {}
    if last.get("error"):
        return "infra:" + last["error"]
    if last.get("stop") == "max_tokens":
        return "truncated"
    if last.get("stop") == "refusal":
        return "refusal"
    return "unparseable"


@contextlib.contextmanager
def observe():
    calls = []
    token = _observed.set(calls)
    try:
        with collect_performance():
            yield calls
    finally:
        _observed.reset(token)


def usage_fields(calls):
    llm = performance_snapshot().get("llm", {})
    return {"llm_calls": llm.get("calls", 0), "input_tokens": llm.get("input_tokens", 0),
            "output_tokens": llm.get("output_tokens", 0),
            "served": sorted({c["served"] for c in calls if c.get("served")}),
            "mismatched": sorted({c["served"] for c in calls if c.get("served") and not c["served"].startswith(c["requested"] or "")})}


def model_mismatches(rows):
    """实际响应模型与请求不一致（供应商降级或改道）时，分数不代表该模型。主模型与辅助模型各按自己的请求核对。"""
    return sorted({m for r in rows for m in r.get("mismatched", [])})


def model_usage():
    """每个模型的输入、输出 token，用于按各自单价估算费用。"""
    totals = {}
    for call in performance_snapshot().get("llm_calls", []):
        row = totals.setdefault(call["model"], [0, 0])
        row[0] += call.get("input_tokens") or 0
        row[1] += call.get("output_tokens") or 0
    return totals


# ── 意图与路由 ────────────────────────────────────────────────────────────────

class BenchmarkRecognizer(IntentRecognizer):
    """生产识别器加观测。offline 让 LLM 分支失败，走真实的降级路径。"""

    def __init__(self, *args, offline=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._offline = offline
        self.client = SpyClient(self.client)

    async def _llm_recognize(self, message, history):
        if self._offline:
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "offline", "failed": True}
        result = await super()._llm_recognize(message, history)
        calls = _observed.get()
        if calls is not None:
            failed = bool(result.get("failed"))
            urgency = result.get("urgency")
            calls.append({"llm": llm_status(failed, calls), "raw": None if failed else [
                result["intent"].value, result["confidence"], urgency.name.lower() if urgency else None]})
        return result


def route(orchestrator, case, intent, urgency, confidence, entities):
    """调用真实的澄清与路由逻辑；澄清由 general 回复。"""
    req = Request(message=case["message"], user_id="benchmark", conv_id=case["id"], intent=intent,
                  intent_group=IntentRecognizer._intent_group(intent), urgency=urgency,
                  intent_confidence=confidence, entities=entities)
    if orchestrator._needs_clarification(req):
        return ["general"]
    return [agent.value for agent in orchestrator._route_decision(req).agent_types]


def route_ok(case, agents):
    if case["agents"] is None:
        return None
    return sorted(agents) in [sorted(option) for option in case["agents"]]


def _norm(value):
    return re.sub(r"\s+", "", value).lower().replace("￥", "¥")


def entity_diff(case, got):
    """返回各实体类型的 (tp, fp, fn)。"""
    types = ALWAYS_SCORED + (("date",) if "date" in case["entities"] else ())
    result = {}
    for kind in types:
        want = {_norm(v) for v in case["entities"].get(kind, [])}
        have = {_norm(v) for v in got.get(kind, [])}
        result[kind] = (len(want & have), len(have - want), len(want - have))
    return result


def score_intent(case, *, intent, confidence, entities, critical, agents, agents_gold):
    gold = IntentCategory(case["intent"])
    diff = entity_diff(case, entities)
    group = IntentRecognizer._intent_group
    return {
        "id": case["id"], "split": case["split"], "tags": case["tags"], "gold": case["intent"],
        "intent": intent.value, "confidence": round(confidence, 4),
        "ok": intent == gold,
        "ok_lenient": intent.value in [case["intent"], *case.get("accept", [])],
        "ok_group": group(intent) == group(gold),
        "entities": {k: v for k, v in entities.items() if v}, "entity_diff": diff,
        "entity_ok": all(fp == fn == 0 for _, fp, fn in diff.values()),
        "critical": critical, "critical_ok": None if case["critical"] is None else critical == case["critical"],
        "agents": agents, "route_ok": route_ok(case, agents),
        "agents_gold": agents_gold, "route_gold_ok": route_ok(case, agents_gold),
    }


async def run_intent(cases, mode, *, reps=1, concurrency=4, recognizer=None, orchestrator=None):
    orchestrator = orchestrator or AgentOrchestrator(api_key="benchmark-router-only")
    majority = collections.Counter(c["intent"] for c in cases).most_common(1)[0][0] if cases else "query"
    gate = asyncio.Semaphore(concurrency)

    async def one(case, rep):
        gold = IntentCategory(case["intent"])
        # 给定标准意图、紧急度和实体，单独衡量路由层
        gold_urgency = UrgencyLevel.CRITICAL if case["critical"] else UrgencyLevel.LOW
        agents_gold = route(orchestrator, case, gold, gold_urgency, 1.0, case["entities"])
        if mode == "oracle":
            fixed = (case["agents"] or [["general"]])[0]
            row = score_intent(case, intent=gold, confidence=1.0, entities=case["entities"],
                               critical=bool(case["critical"]), agents=fixed, agents_gold=fixed)
            return {**row, "rep": rep, "llm": "none"}
        if mode == "majority":
            row = score_intent(case, intent=IntentCategory(majority), confidence=1.0, entities={},
                               critical=False, agents=["general"], agents_gold=agents_gold)
            return {**row, "rep": rep, "llm": "none"}
        async with gate:
            with observe() as calls:
                result = await recognizer.recognize(case["message"], history=case.get("history"))
                usage = usage_fields(calls)
        branch = next((c for c in calls if "llm" in c), {"llm": "offline"})
        agents = route(orchestrator, case, result.intent, result.urgency, result.confidence, result.entities)
        row = score_intent(case, intent=result.intent, confidence=result.confidence, entities=result.entities,
                           critical=result.urgency == UrgencyLevel.CRITICAL, agents=agents, agents_gold=agents_gold)
        return {**row, "rep": rep, "llm": branch["llm"], "llm_raw": branch.get("raw"), "embedding": result.embedding_source,
                "latency_ms": round(result.latency_ms, 1), **usage}

    rows = []
    for rep in range(reps):
        if recognizer is not None:      # 每轮独立，避免命中上一轮缓存
            recognizer._cache.clear()
            recognizer._cache_expiries.clear()
        rows += await asyncio.gather(*(one(case, rep) for case in cases))
    return rows


def summarize_intent(rows):
    labels = sorted({r["gold"] for r in rows} | {r["intent"] for r in rows})
    per_class = {}
    for label in labels:
        tp = sum(r["gold"] == r["intent"] == label for r in rows)
        fp = sum(r["intent"] == label != r["gold"] for r in rows)
        fn = sum(r["gold"] == label != r["intent"] for r in rows)
        per_class[label] = {"precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0, "n": tp + fn}
    gold_labels = [v for v in per_class.values() if v["n"]]
    entities = {}
    for kind in ALWAYS_SCORED + ("date",):
        diffs = [r["entity_diff"][kind] for r in rows if kind in r["entity_diff"]]
        tp, fp, fn = (sum(d[i] for d in diffs) for i in range(3))
        entities[kind] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                          "recall": round(tp / (tp + fn), 4) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}
    urgent = [r for r in rows if r["critical_ok"] is not None]
    hit = sum(r["critical"] and r["critical_ok"] for r in urgent)
    false = sum(r["critical"] and not r["critical_ok"] for r in urgent)
    missed = sum(not r["critical"] and not r["critical_ok"] for r in urgent)
    routed = [r for r in rows if r["route_ok"] is not None]
    by = lambda key: {name: {"ok": rate(group, "ok")["rate"], "route_ok": rate(group, "route_ok")["rate"], "n": len(group)}
                      for name, group in sorted(_group(rows, key).items())}
    return {
        "accuracy": rate(rows, "ok"), "accuracy_lenient": rate(rows, "ok_lenient"), "accuracy_group": rate(rows, "ok_group"),
        "macro_f1": round(sum(v["f1"] for v in gold_labels) / len(gold_labels), 4) if gold_labels else None,
        "accuracy_llm_ok": rate([r for r in rows if r["llm"] == "ok"], "ok"),
        "route": rate(rows, "route_ok"), "route_given_gold": rate(rows, "route_gold_ok"),
        "route_extra_agents": sum(not r["route_ok"] and len(r["agents"]) > 1 for r in routed),
        "entities_exact": rate(rows, "entity_ok"), "entities": entities,
        "critical": {"precision": round(hit / (hit + false), 4) if hit + false else None,
                     "recall": round(hit / (hit + missed), 4) if hit + missed else None,
                     "true": hit, "false_alarm": false, "missed": missed},
        "llm_status": dict(collections.Counter(r["llm"] for r in rows)),
        "embedding_source": dict(collections.Counter(r["embedding"] for r in rows if "embedding" in r)),
        "noise_floor": round(1 / math.sqrt(len(rows)), 4) if rows else None,
        "by_split": by(lambda r: [r["split"]]), "by_tag": by(lambda r: r["tags"] or ["plain"]),
        "per_class": per_class,
        "confusions": [{"gold": g, "predicted": p, "n": n} for (g, p), n in collections.Counter(
            (r["gold"], r["intent"]) for r in rows if not r["ok_lenient"]).most_common(10)],
        **cost_summary(rows),
    }


def _group(rows, key):
    groups = collections.defaultdict(list)
    for row in rows:
        for name in key(row):
            groups[name].append(row)
    return groups


# ── 检索 ──────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def evaluation_kb(corpus, embedder=None):
    """用生产 KnowledgeBase 在临时目录建库：切片、向量、打分与线上一致，且不接触真实知识库。"""
    from tooling.knowledge_base import KnowledgeBase
    overrides = {"KORA_ENV": "development", "KORA_SEED_DEMO_DOCS": "false"}
    saved = {k: os.environ.get(k) for k in overrides}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as path:
        os.environ.update(overrides)
        try:
            kb = KnowledgeBase(chroma_host="127.0.0.1", chroma_port=1, chroma_path=path, embedder=embedder)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        if kb._use_server or kb.doc_count:
            raise RuntimeError("评测知识库必须是空的本地临时库")
        kb.add_documents([{"title": d["title"], "content": d["content"], "document_id": d["id"]} for d in corpus])
        yield kb


class LexicalIndex:
    """字符二元组 BM25，作为不依赖向量模型的参照。"""

    def __init__(self, corpus, chunk):
        self.items = [{"title": d["title"], "content": text} for d in corpus for text in chunk(d["content"])]
        self.docs = [collections.Counter(self._grams(i["title"] + i["content"])) for i in self.items]
        self.df = collections.Counter(g for doc in self.docs for g in doc)
        self.avg = sum(sum(d.values()) for d in self.docs) / max(1, len(self.docs))

    @staticmethod
    def _grams(text):
        text = re.sub(r"[\W_]+", "", text.lower())
        return [text[i:i + 2] for i in range(len(text) - 1)]

    def search(self, query, top_k):
        n, scored = len(self.docs), []
        for item, doc in zip(self.items, self.docs):
            size, score = sum(doc.values()), 0.0
            for gram in set(self._grams(query)):
                if tf := doc.get(gram):
                    idf = math.log(1 + (n - self.df[gram] + 0.5) / (self.df[gram] + 0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * size / self.avg))
            if score:
                scored.append({**item, "score": round(score, 4)})
        return sorted(scored, key=lambda i: -i["score"])[:top_k]


def score_retrieval(case, items, titles):
    """hit/recall 按文档，answer5 要求答案片段全部出现在返回的切片里。"""
    ranked = [titles.get(i.get("title")) for i in items[:TOP_K]]
    row = {"id": case["id"], "split": case["split"], "tags": case["tags"], "returned": len(items), "ranked": ranked,
           "top_score": max((i["score"] for i in items if i.get("score") is not None), default=None)}
    if not case["relevant"]:
        return {**row, "answerable": False}
    first = next((n for n, doc in enumerate(ranked, 1) if doc in case["relevant"]), None)
    text = "\n".join(i.get("content", "") for i in items[:TOP_K])
    return {**row, "answerable": True, "hit1": first == 1, "hit3": first is not None and first <= 3, "hit5": first is not None,
            "recall5": round(len(set(ranked) & set(case["relevant"])) / len(case["relevant"]), 4),
            "mrr5": round(1 / first, 4) if first else 0.0, "answer5": all(span in text for span in case["answer"])}


async def run_retrieval(cases, corpus, search, *, reps=1, concurrency=4):
    """search: async (query, top_k) -> (items, error)。检索报错单独记录，不当作没召回。"""
    titles = {d["title"]: d["id"] for d in corpus}
    gate = asyncio.Semaphore(concurrency)

    async def one(case, rep):
        async with gate:
            with observe() as calls:
                started = time.monotonic()
                items, error = await search(case, TOP_K)
                latency = (time.monotonic() - started) * 1000
                usage = usage_fields(calls)
        failed = [c["error"] for c in calls if c.get("error")]
        base = {"rep": rep, "latency_ms": round(latency, 1), "llm_errors": failed, **usage}
        if error:
            return {"id": case["id"], "split": case["split"], "tags": case["tags"], "error": error,
                    "answerable": bool(case["relevant"]), **base}
        return {**score_retrieval(case, items, titles), **base}

    rows = []
    for rep in range(reps):
        rows += await asyncio.gather(*(one(case, rep) for case in cases))
    return rows


def summarize_retrieval(rows):
    valid = [r for r in rows if "error" not in r]
    yes, no = [r for r in valid if r["answerable"]], [r for r in valid if not r["answerable"]]
    mean = lambda key: round(sum(r[key] for r in yes) / len(yes), 4) if yes else None
    scores = lambda group: [r["top_score"] for r in group if r["top_score"] is not None]
    return {
        "hit1": rate(yes, "hit1"), "hit3": rate(yes, "hit3"), "hit5": rate(yes, "hit5"), "answer5": rate(yes, "answer5"),
        "recall5": mean("recall5"), "mrr5": mean("mrr5"),
        "unanswerable": {"n": len(no), "returned_any": sum(r["returned"] > 0 for r in no),
                         "top_score_auc": auc(scores(yes), scores(no)),
                         "top_score_p50": {"answerable": pct(scores(yes), 50, 4), "unanswerable": pct(scores(no), 50, 4)}},
        "errors": dict(collections.Counter(r["error"] for r in rows if "error" in r)),
        "llm_errors": dict(collections.Counter(e for r in rows for e in r["llm_errors"])),
        "by_split": {name: {"hit1": rate(g, "hit1")["rate"], "hit5": rate(g, "hit5")["rate"], "n": len(g)}
                     for name, g in sorted(_group(yes, lambda r: [r["split"]]).items())},
        "by_tag": {name: {"hit1": rate(g, "hit1")["rate"], "hit5": rate(g, "hit5")["rate"],
                          "answer5": rate(g, "answer5")["rate"], "n": len(g)}
                   for name, g in sorted(_group(yes, lambda r: r["tags"]).items())},
        **cost_summary(valid),
    }


def tool_manager_for(kb, cfg):
    """与 api/main.py 相同的 knowledge_search 注册方式。"""
    from tooling.tool_manager import MCPToolManager, Tool
    manager = MCPToolManager(**cfg)
    manager._client = SpyClient(manager._client)
    manager.register(Tool(name="knowledge_search", description="搜索知识库（基于 ChromaDB 向量检索）", handler=kb.search_handler,
                          schema={"type": "object", "properties": {
                              "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                              "top_k": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query"]},
                          cache_ttl=300.0, supports_rerank=True))
    return manager


# ── 端到端耗时 ────────────────────────────────────────────────────────────────

async def run_e2e(cases, orchestrator, *, reps=1, concurrency=1, timeout=50, stream=False):
    """编排层完整一轮（不含 HTTP 与记忆），只看耗时、调用次数、token 和执行状态，不评价回答质量。"""
    gate = asyncio.Semaphore(concurrency)

    async def one(case, rep):
        history = case.get("history")
        context = "[最近对话]\n" + "\n".join(f"{m['role']}: {m['content']}" for m in history) if history else ""
        row = {"id": case["id"], "split": case["split"], "tags": case["tags"], "rep": rep}
        async with gate:
            with observe() as calls:
                started = time.monotonic()
                emitter = Emitter() if stream else None
                req = Request(message=case["message"], user_id="benchmark", conv_id=f"{case['id']}-{rep}",
                              context=context, history=history, conversation_state={}, emit=emitter)
                try:
                    result = await asyncio.wait_for(orchestrator.run(req), timeout)
                    row.update(status=result.status, agents=[a.value for a in result.agent_types],
                               tools=result.tools_used, answer_chars=len(result.response),
                               prefetched=sum(t.get("results", 0) for t in getattr(result, "tool_traces", []) if t.get("event") == "knowledge_prefetch"))
                    if emitter:
                        streamed = emitter.text
                        emitter.finish(result.response)
                        row["first_token_ms"] = emitter.first_ms
                        # 流出的正文与最终回复不一致时 finish 会整段重发，首字耗时随之失真；记下事件序列便于排查。
                        a, b = streamed.strip(), result.response.strip()
                        row["stream"] = {"matched": a == b, "streamed_chars": len(a), "final_chars": len(b),
                                         "common_prefix": len(os.path.commonprefix([a, b])),
                                         "common_suffix": len(os.path.commonprefix([a[::-1], b[::-1]])),
                                         "events": stream_events(emitter)}
                except asyncio.TimeoutError:
                    row.update(status="timeout", agents=[], tools=[])
                except Exception as ex:     # 单例异常不应丢掉已付费的其他结果
                    row.update(status=f"error:{type(ex).__name__}", agents=[], tools=[])
                row.update(latency_ms=round((time.monotonic() - started) * 1000, 1), **usage_fields(calls))
                row["stages"] = dict(collections.Counter(c["stage"] for c in performance_snapshot()["llm_calls"]))
                row["tokens_by_model"] = model_usage()
        row["llm_errors"] = [c["error"] for c in calls if c.get("error")]
        row["llm_blocks"] = [c["blocks"] for c in calls if c.get("blocks")]
        row["success"] = row["status"] == "success"
        return row

    rows = []
    for rep in range(reps):
        orchestrator._intent_recognizer._cache.clear()      # 每轮独立，避免命中上一轮缓存
        rows += await asyncio.gather(*(one(case, rep) for case in cases))
    return rows


def stream_events(emitter):
    """未被消费的事件按顺序压缩成 "status:answer delta×12 reset ..."，不含正文。"""
    out = []
    for event, data in list(emitter._queue._queue):
        name = f"{event}:{data['stage']}" if event == "status" else event
        if out and out[-1][0] == name:
            out[-1][1] += 1
        else:
            out.append([name, 1])
    return " ".join(name if n == 1 else f"{name}×{n}" for name, n in out)


def summarize_e2e(rows):
    first = [r["first_token_ms"] for r in rows if r.get("first_token_ms") is not None]
    tokens = {}
    for row in rows:
        for model, (given, produced) in row.get("tokens_by_model", {}).items():
            total = tokens.setdefault(model, [0, 0])
            total[0], total[1] = total[0] + given, total[1] + produced
    return {"success": rate(rows, "success"), "status": dict(collections.Counter(r["status"] for r in rows)),
            "agents": dict(collections.Counter("+".join(sorted(r["agents"])) or "none" for r in rows)),
            "tool_calls": dict(collections.Counter(t for r in rows for t in r["tools"])),
            "llm_errors": dict(collections.Counter(e for r in rows for e in r["llm_errors"])),
            "llm_calls_by_stage": dict(sum((collections.Counter(r["stages"]) for r in rows), collections.Counter())),
            "latency_by_agent_count": {str(n): {"p50": pct([r["latency_ms"] for r in g], 50), "n": len(g)}
                                       for n, g in sorted(_group(rows, lambda r: [len(r["agents"])]).items())},
            "first_token_ms": {"p50": pct(first, 50), "p95": pct(first, 95), "n": len(first)},
            "prefetched_turns": sum(r.get("prefetched", 0) > 0 for r in rows),
            "stream_mismatch": [r["id"] for r in rows if not r.get("stream", {}).get("matched", True)],
            "tokens_by_model": {model: {"input": given, "output": produced} for model, (given, produced) in tokens.items()},
            **cost_summary(rows)}


# ── 报告与对比 ────────────────────────────────────────────────────────────────

def report_status(suite, rows, mismatches):
    """partial 表示含基础设施失败或模型不符，数字只能参考，不应作为基线。"""
    if not rows:
        return "empty"
    infra = any(r.get("llm", "").startswith("infra:") or r.get("llm_errors") or r.get("error")
                or str(r.get("status", "")).startswith(("timeout", "error:"))
                for r in rows)
    return "partial" if infra or mismatches else "complete"


def build_report(suite, mode, split, reps, dataset_id, rows, config):
    metrics = {"intent": summarize_intent, "retrieval": summarize_retrieval, "e2e": summarize_e2e}[suite](rows)
    mismatches = model_mismatches(rows)
    return {"suite": suite, "mode": mode, "split": split, "reps": reps, "cases": len({r["id"] for r in rows}),
            "dataset": dataset_id, "status": report_status(suite, rows, mismatches), "model_mismatches": mismatches,
            "timestamp": datetime.now().isoformat(timespec="seconds"), "git": git_revision(), "config": config,
            "metrics": metrics, "rows": rows}


def git_revision():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain", "--", "agents", "core", "tooling") else "")
    except Exception:
        return None


def save_report(report, path):
    """明细每例一行，便于 diff。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2)
    body = ",\n".join("    " + json.dumps(r, ensure_ascii=False) for r in report["rows"])
    path.write_text(head[:-2] + ',\n  "rows": [\n' + body + "\n  ]\n}\n", encoding="utf-8", newline="\n")


def per_case(report, key):
    """多轮取多数，得到逐用例的通过与否。"""
    values = collections.defaultdict(list)
    for row in report["rows"]:
        if row.get(key) is not None:
            values[row["id"]].append(bool(row[key]))
    return {case: sum(v) * 2 > len(v) for case, v in values.items()}


def compare(baseline, current):
    """同一数据集逐用例配对比较；数据集变了就拒绝比较。"""
    for key in ("suite", "dataset"):
        if baseline[key] != current[key]:
            raise ValueError(f"{key} 不同，不能比较：{baseline[key]} / {current[key]}")
    result = {"partial": [name for name, r in (("baseline", baseline), ("current", current)) if r["status"] != "complete"]}
    for key in BINARY[current["suite"]]:
        before, after = per_case(baseline, key), per_case(current, key)
        shared = sorted(before.keys() & after.keys())
        if not shared:
            continue
        fixed = [c for c in shared if after[c] and not before[c]]
        broken = [c for c in shared if before[c] and not after[c]]
        result[key] = {"n": len(shared), "before": round(sum(before[c] for c in shared) / len(shared), 4),
                       "after": round(sum(after[c] for c in shared) / len(shared), 4),
                       "fixed": fixed, "broken": broken, "p": mcnemar(len(fixed), len(broken))}
    return result


def _fmt(value):
    if isinstance(value, dict) and "rate" in value:
        if value["rate"] is None:
            return "—"
        low, high = value["ci95"]
        return f"{value['rate']:.1%}  (n={value['n']}, 95% CI {low:.1%}–{high:.1%})"
    return json.dumps(value, ensure_ascii=False)


def print_report(report):
    m = report["metrics"]
    print(f"\n[{report['suite']} / {report['mode']} / {report['split']}] 用例 {report['cases']} × {report['reps']} 轮  "
          f"状态 {report['status']}  数据集 {report['dataset']}  代码 {report['git']}")
    if report["model_mismatches"]:
        print("  ⚠ 实际响应模型与请求不一致:", report["model_mismatches"])
    hidden = {"per_class", "by_tag", "by_split", "confusions"}
    for key, value in m.items():
        if key not in hidden:
            print(f"  {key:20} {_fmt(value)}")
    for title in ("by_split", "by_tag"):
        for name, group in m.get(title, {}).items():
            print(f"  {title}:{name:16} " + "  ".join(f"{k}={v:.0%}" if isinstance(v, float) else f"{k}={v}" for k, v in group.items()))
    for c in m.get("confusions", []):
        print(f"  误判 {c['gold']} → {c['predicted']} × {c['n']}")


def print_comparison(result):
    print("\n与基线对比（逐用例配对）")
    if result["partial"]:
        print("  ⚠ 含未完整的报告:", result["partial"])
    for key, r in result.items():
        if key == "partial":
            continue
        verdict = "差异显著" if r["p"] < 0.05 else "未超出噪声"
        print(f"  {key:14} {r['before']:.1%} → {r['after']:.1%}  修好 {len(r['fixed'])}  弄坏 {len(r['broken'])}  p{'<0.0001' if r['p'] < 0.0001 else '=' + str(r['p'])}  {verdict}")
        if r["broken"]:
            print(f"    弄坏: {', '.join(r['broken'][:15])}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def live_config(model=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise SystemExit("live 模式需要在 .env 配置 ANTHROPIC_API_KEY")
    cfg = {"api_key": key, "model": model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()}
    if base_url := os.getenv("ANTHROPIC_BASE_URL", "").strip():
        cfg["base_url"] = base_url
    return cfg


async def execute(args):
    intents, corpus, queries = load_jsonl("intent.jsonl"), load_jsonl("retrieval_corpus.jsonl"), load_jsonl("retrieval.jsonl")
    live = args.mode == "live"
    cfg = live_config(args.model) if live else {}
    if live and not args.model and args.suite != "e2e":     # 与生产一致：意图和检索里的模型调用用辅助模型
        cfg["model"] = fast_model(cfg["model"])
    embedder = None
    if args.mode in ("offline", "vector", "live"):      # 与生产一致：启动时加载本地向量模型
        from core.embedding import load_embedder
        embedder = load_embedder()
    config = {"model": cfg.get("model"), "fast_model": fast_model(cfg["model"]) if live else None,
              "embedding": embedder.name if embedder else "chroma-default", "concurrency": args.concurrency, "limit": args.limit}

    if args.suite == "intent":
        cases = select(intents, args.split, args.limit, lambda r: r["intent"])
        if live:
            print(f"将调用模型约 {len(cases) * args.reps} 次（{cfg['model']}）")
            recognizer = BenchmarkRecognizer(**cfg)
        elif args.mode == "offline":
            os.environ.update(INTENT_EMBEDDING_PROVIDER="local", INTENT_EMBEDDING_ENABLED="true")
            recognizer = BenchmarkRecognizer(api_key="offline", offline=True)
        else:
            recognizer = None
        rows = await run_intent(cases, args.mode, reps=args.reps, concurrency=args.concurrency, recognizer=recognizer)
        return build_report("intent", args.mode, args.split, args.reps, fingerprint(intents), rows, config)

    if args.suite == "retrieval":
        if args.lang == "en":
            queries = load_jsonl("retrieval_en.jsonl")
        config["lang"] = args.lang
        cases = select(queries, args.split, args.limit, lambda r: r["tags"][-1])
        dataset_id = fingerprint(corpus, queries)
        by_id = {d["id"]: d for d in corpus}

        async def run(search):
            rows = await run_retrieval(cases, corpus, search, reps=args.reps, concurrency=args.concurrency)
            return build_report("retrieval", args.mode, args.split, args.reps, dataset_id, rows, config)

        if args.mode == "oracle":
            async def oracle(case, top_k):
                return [{"title": by_id[i]["title"], "content": by_id[i]["content"], "score": 1.0} for i in case["relevant"]], None
            return await run(oracle)
        if args.mode == "null":
            async def null(case, top_k):
                return [], None
            return await run(null)
        if args.mode == "lexical":
            from tooling.knowledge_base import KnowledgeBase
            index = LexicalIndex(corpus, lambda text: KnowledgeBase._chunk_text(None, text))

            async def lexical(case, top_k):
                return index.search(case["query"], top_k), None
            return await run(lexical)
        with evaluation_kb(corpus, embedder) as kb:
            config["chunks"] = kb.doc_count
            if live:
                print(f"将调用模型约 {len(cases) * args.reps}–{len(cases) * args.reps * 2} 次（{cfg['model']}）")
                manager = tool_manager_for(kb, cfg)

                async def pipeline(case, top_k):
                    result = await manager.search_with_rewrite("knowledge_search", case["query"], top_k=top_k)
                    return result.data or [], None if result.success else result.error
                return await run(pipeline)

            async def vector(case, top_k):
                return await kb.search_async(case["query"], top_k=top_k), None
            return await run(vector)

    cases = select(intents, args.split, args.limit or 30, lambda r: r["intent"])
    if args.ids:
        wanted = args.ids.split(",")
        cases = [case for case in intents if case["id"] in wanted]
    print(f"将执行 {len(cases) * args.reps} 轮完整对话，每轮约 2–10 次模型调用（{cfg['model']}）")
    from agents.tools import build_shared_rag_tools
    from core.skill_loader import SkillManager
    skills = SkillManager(root_dir=os.getenv("KORA_SKILLS_DIR") or str(ROOT / "skills"))
    skills.load()
    orchestrator = AgentOrchestrator(**cfg, skill_manager=skills)
    spy = SpyClient(orchestrator._composer._client)
    orchestrator._composer._client = spy
    for agents in orchestrator._pool.values():
        for agent in agents:
            agent._client = spy
    orchestrator._intent_recognizer.client = SpyClient(orchestrator._intent_recognizer.client)
    with evaluation_kb(corpus, embedder) as kb:
        manager = tool_manager_for(kb, cfg)
        manager.rag_rewrite = os.getenv("KORA_RAG_REWRITE", "false").lower() in {"1", "true", "yes", "on"}
        manager.rag_rerank = os.getenv("KORA_RAG_RERANK", "false").lower() in {"1", "true", "yes", "on"}
        config.update(rag_rewrite=manager.rag_rewrite, rag_rerank=manager.rag_rerank)
        orchestrator.set_shared_tools(build_shared_rag_tools(manager))
        prefetch = os.getenv("KORA_PREFETCH_KNOWLEDGE", "true").lower() in {"1", "true", "yes", "on"}
        if prefetch:
            orchestrator.set_knowledge_prefetch(lambda query: manager.call("knowledge_search", {"query": query, "top_k": 5}))
        config.update(prefetch=prefetch, stream=args.stream)
        rows = await run_e2e(cases, orchestrator, reps=args.reps, concurrency=args.concurrency, stream=args.stream)
    return build_report("e2e", "live", args.split, args.reps, fingerprint(intents, corpus), rows, config)


def main(argv=None):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m evaluation.benchmark", description=__doc__.split("\n")[0])
    parser.add_argument("suite", choices=list(MODES))
    parser.add_argument("--mode", help="intent: %s；retrieval: %s；e2e: live" % ("/".join(MODES["intent"]), "/".join(MODES["retrieval"])))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh", help="retrieval 使用的查询集")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（各类别轮流抽取）")
    parser.add_argument("--ids", help="e2e 只跑这些用例，逗号分隔")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=0, help="默认 intent/retrieval 为 4，e2e 为 1")
    parser.add_argument("--model", help="临时覆盖 ANTHROPIC_MODEL")
    parser.add_argument("--stream", action="store_true", help="e2e 使用流式输出并记录首字耗时")
    parser.add_argument("--out")
    parser.add_argument("--compare", help="与此前的报告逐用例比较")
    args = parser.parse_args(argv)
    args.mode = args.mode or MODES[args.suite][0]
    if args.mode not in MODES[args.suite]:
        parser.error(f"{args.suite} 不支持 --mode {args.mode}，可选 {MODES[args.suite]}")
    args.concurrency = args.concurrency or (1 if args.suite == "e2e" else 4)
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

    report = asyncio.run(execute(args))
    out = args.out or ROOT / "data" / "eval" / f"benchmark-{args.suite}-{args.mode}-{args.split}-{datetime.now():%Y%m%d-%H%M%S}.json"
    save_report(report, out)
    print_report(report)
    print(f"\n报告已保存: {out}")
    if args.compare:
        print_comparison(compare(json.loads(pathlib.Path(args.compare).read_text(encoding="utf-8")), report))


if __name__ == "__main__":
    main()
