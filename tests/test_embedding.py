import asyncio
import hashlib
import io
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

try:
    import chromadb
except ModuleNotFoundError:
    sys.modules['chromadb'] = types.ModuleType('chromadb')

from core import embedding
from core.intent_embedding import IntentEmbedding
from core.intent_recognizer import IntentCategory, IntentRecognizer
from tooling.knowledge_base import KnowledgeBase
from tooling.tool_manager import MCPToolManager, Tool

real_chroma = pytest.mark.skipif(not hasattr(sys.modules['chromadb'], 'PersistentClient'), reason='需要 chromadb')


class FakeEmbedder:
    """按关键词给出确定的二维向量。"""
    name = 'fake'

    def __init__(self):
        self.texts = []

    def embed(self, texts):
        self.texts += texts
        return [[1.0, 0.0] if '退款' in t or 'refund' in t.lower() else [0.0, 1.0] for t in texts]


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True))


@pytest.fixture(autouse=True)
def no_global_model(monkeypatch):
    monkeypatch.setattr(embedding, '_current', None)


# ── 集合与迁移 ────────────────────────────────────────────────────────────────

@real_chroma
def test_collection_embeds_with_title_and_returns_cosine_distance(client):
    model = FakeEmbedder()
    target = embedding.open_collection(client, 'docs', model, text_of=lambda d, m: f"{m['title']}\n{d}", metadata={'description': 'x'})
    target.upsert(ids=['a', 'b'], documents=['七天内', '三到五天'], metadatas=[{'title': '退款'}, {'title': '配送'}])
    assert model.texts == ['退款\n七天内', '配送\n三到五天']
    result = target.query(query_texts=['refund please'], n_results=2)
    assert result['ids'][0] == ['a', 'b']
    assert result['distances'][0] == pytest.approx([0.0, 1.0], abs=1e-5)


@real_chroma
def test_migration_runs_once_keeps_old_collection_and_never_restores_deleted_rows(client):
    old = client.get_or_create_collection('docs', embedding_function=None)
    old.upsert(ids=[str(i) for i in range(130)], documents=[f'退款 {i}' for i in range(130)],
               metadatas=[{'title': 't'}] * 130, embeddings=[[0.1, 0.2, 0.3]] * 130)
    target = embedding.open_collection(client, 'docs', FakeEmbedder())
    assert target.count() == old.count() == 130
    target.delete(ids=['0'])
    model = FakeEmbedder()
    again = embedding.open_collection(client, 'docs', model)
    assert again.count() == 129 and model.texts == [] and old.count() == 130


@real_chroma
def test_interrupted_migration_resumes_next_time(client, monkeypatch):
    old = client.get_or_create_collection('docs', embedding_function=None)
    old.upsert(ids=['1', '2'], documents=['退款', '配送'], metadatas=[{'t': 1}] * 2, embeddings=[[0.1, 0.2, 0.3]] * 2)
    broken = FakeEmbedder()
    broken.embed = lambda texts: (_ for _ in ()).throw(RuntimeError('crash'))
    with pytest.raises(RuntimeError, match='crash'):
        embedding.open_collection(client, 'docs', broken)
    assert embedding.open_collection(client, 'docs', FakeEmbedder()).count() == 2


def test_listing_failure_is_not_treated_as_missing_collection():
    client = SimpleNamespace(get_or_create_collection=lambda **k: object(),
                             list_collections=lambda: (_ for _ in ()).throw(ConnectionError('down')))
    with pytest.raises(ConnectionError):
        embedding.open_collection(client, 'docs', FakeEmbedder())


def test_without_model_the_original_collection_is_used():
    calls = []
    client = SimpleNamespace(get_or_create_collection=lambda **k: calls.append(k) or 'raw')
    assert embedding.open_collection(client, 'docs', None, metadata={'d': 1}) == 'raw'
    assert embedding.open_collection(client, 'episodic', None) == 'raw'
    assert calls == [{'name': 'docs', 'metadata': {'d': 1}}, {'name': 'episodic'}]


# ── 加载与下载 ────────────────────────────────────────────────────────────────

def test_missing_model_fails_clearly_in_production_instead_of_falling_back(monkeypatch, tmp_path):
    monkeypatch.setenv('KORA_ENV', 'production')
    monkeypatch.setenv('KORA_EMBEDDING_DIR', str(tmp_path))
    monkeypatch.delenv('KORA_EMBEDDING_DOWNLOAD', raising=False)
    monkeypatch.setattr(embedding.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('生产环境不应自动下载'))
    with pytest.raises(RuntimeError, match='缺少向量模型文件'):
        embedding.load_embedder()


def test_provider_switch(monkeypatch):
    monkeypatch.setenv('KORA_EMBEDDING_PROVIDER', 'chroma')
    assert embedding.load_embedder() is None and embedding.current() is None
    monkeypatch.setenv('KORA_EMBEDDING_PROVIDER', 'other')
    with pytest.raises(ValueError):
        embedding.load_embedder()


def test_download_is_pinned_verified_and_atomic(monkeypatch, tmp_path):
    good, urls = b'model-bytes', []

    def urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(good if 'good' in url else b'tampered')

    monkeypatch.setenv('HF_ENDPOINT', 'https://mirror.example/')
    monkeypatch.setattr(embedding.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(embedding, 'FILES', {'a': ('good', hashlib.sha256(good).hexdigest())})
    embedding.ensure_files(tmp_path)
    embedding.ensure_files(tmp_path)
    assert (tmp_path / 'a').read_bytes() == good
    assert urls == [f'https://mirror.example/{embedding.REPO}/resolve/{embedding.REVISION}/good']
    monkeypatch.setattr(embedding, 'FILES', {'b': ('bad', hashlib.sha256(good).hexdigest())})
    with pytest.raises(RuntimeError, match='校验失败'):
        embedding.ensure_files(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a']


# ── 知识库阈值与检索开关 ──────────────────────────────────────────────────────

def kb_with(distances, min_score):
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._min_score = min_score
    kb._collection = SimpleNamespace(count=lambda: len(distances), query=lambda **k: {
        'documents': [[f'd{i}' for i in range(len(distances))]], 'metadatas': [[{'title': 't'}] * len(distances)], 'distances': [distances]})
    return kb


def test_low_scores_are_dropped_only_when_a_threshold_is_set():
    assert [i['score'] for i in kb_with([0.3, 0.6, 0.7], 0.4).search('q')] == [0.7, 0.4]
    assert len(kb_with([0.3, 1.6], None).search('q')) == 2


def manager_with(handler, **flags):
    manager = MCPToolManager('test')
    manager.register(Tool('search', '', handler, {}, cache_ttl=30, supports_rerank=True))
    manager.rewrite_query = AsyncMock(return_value=['q', 'q2'])
    manager._rerank_with_status = AsyncMock(return_value=(['reranked'], True))
    for key, value in flags.items():
        setattr(manager, key, value)
    return manager


def test_rewrite_and_rerank_can_be_switched_off_independently():
    async def handler(params, context):
        return [{'content': params['query'] + '-low', 'score': 0.4}, {'content': params['query'] + '-high', 'score': 0.8}]

    async def run():
        off = manager_with(handler, rag_rewrite=False, rag_rerank=False)
        result = await off.search_with_rewrite('search', 'q', top_k=1)
        assert result.success and not result.reranked and result.data == [{'content': 'q-high', 'score': 0.8}]
        off.rewrite_query.assert_not_awaited()
        off._rerank_with_status.assert_not_awaited()
        fork = off.fork()
        assert (fork.rag_rewrite, fork.rag_rerank) == (False, False)
        rewrite = manager_with(handler, rag_rerank=False)
        result = await rewrite.search_with_rewrite('search', 'q', top_k=3)
        assert [i['score'] for i in result.data] == [0.8, 0.8, 0.4]
        assert (await manager_with(handler, rag_rewrite=False).search_with_rewrite('search', 'q')).data == ['reranked']
        for item in (off, rewrite):
            await item._client.close()
    asyncio.run(run())


def test_no_match_is_success_but_failure_is_not():
    from agents.tools import build_shared_rag_tools

    async def empty(params, context):
        return []

    async def broken(params, context):
        raise RuntimeError('down')

    async def run():
        manager = manager_with(empty, rag_rewrite=False)
        result = await manager.search_with_rewrite('search', 'q')
        assert result.success and result.data == [] and not result.error
        failed = await manager_with(broken, rag_rewrite=False).search_with_rewrite('search', 'q')
        assert not failed.success and 'down' in failed.error
        manager._tools['knowledge_search'] = manager._tools.pop('search')
        reply = await build_shared_rag_tools(manager)['search_knowledge_base'].handler(SimpleNamespace(message='q'), {'query': 'q'})
        assert reply['success'] and reply['results'] == [] and '没有找到' in reply['message']
        await manager._client.close()
    asyncio.run(run())


# ── 意图向量 ──────────────────────────────────────────────────────────────────

def test_intent_prefers_loaded_model_then_falls_back_to_ngram(monkeypatch):
    monkeypatch.setenv('INTENT_EMBEDDING_PROVIDER', 'local')
    model, templates = FakeEmbedder(), {'refund': ['我要退款'], 'other': ['配送多久', '在吗']}

    async def run():
        branch = IntentEmbedding()
        monkeypatch.setattr(embedding, '_current', model)
        groups, query, source = await branch.vectors('refund', templates, lambda t: [9.0])
        assert source == 'local-fake' and query == [1.0, 0.0]
        assert {k: len(v) for k, v in groups.items()} == {'refund': 1, 'other': 2}
        await branch.vectors('again', templates, lambda t: [9.0])
        assert model.texts == ['我要退款', '配送多久', '在吗', 'refund', 'again']     # 模板只算一次
        model.embed = lambda texts: (_ for _ in ()).throw(RuntimeError('broken'))
        _, query, source = await branch.vectors('x', templates, lambda t: [9.0])
        assert source == 'local-ngram' and query == [9.0]
    asyncio.run(run())


@pytest.mark.parametrize('source,confidence,expected', [
    ('local-bge-m3-int8', 0.59, IntentCategory.OTHER), ('local-bge-m3-int8', 0.61, IntentCategory.REFUND),
    ('siliconflow/BAAI/bge-m3', 0.55, IntentCategory.OTHER), ('local-ngram', 0.55, IntentCategory.REFUND)])
def test_failed_llm_uses_a_threshold_matching_the_vector_source(source, confidence, expected):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled, recognizer.threshold = True, 0.5
    failed = {'intent': IntentCategory.OTHER, 'confidence': 0.0, 'failed': True}
    nothing = {'intent': IntentCategory.OTHER, 'confidence': 0.0}
    vector = {'intent': IntentCategory.REFUND, 'confidence': confidence, 'source': source}
    assert recognizer._vote(failed, vector, nothing)[0] is expected
