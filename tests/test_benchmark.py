import asyncio
import json
import os
import re
import sys
import types
from collections import Counter
from types import SimpleNamespace

import pytest

try:
    import chromadb
except ModuleNotFoundError:
    sys.modules['chromadb'] = types.ModuleType('chromadb')

from core.intent_recognizer import _TEMPLATES, IntentCategory
from evaluation import benchmark as bench
from evaluation.evaluator import DEFAULT_INTENT_CASES

INTENTS = bench.load_jsonl('intent.jsonl')
CORPUS = bench.load_jsonl('retrieval_corpus.jsonl')
QUERIES = bench.load_jsonl('retrieval.jsonl')
LABELS = {c.value for c in IntentCategory}
AGENTS = {'general', 'technical', 'billing', 'escalation'}


def grams(text):
    text = re.sub(r'[\W_]+', '', text.lower())
    return {text[i:i + 2] for i in range(len(text) - 1)} or {text}


# ── 数据集 ────────────────────────────────────────────────────────────────────

def test_intent_cases_follow_schema_and_cover_every_label_in_both_splits():
    assert len({c['id'] for c in INTENTS}) == len(INTENTS) >= 150
    assert len({(c['message'], json.dumps(c.get('history'))) for c in INTENTS}) == len(INTENTS)
    for case in INTENTS:
        assert case['intent'] in LABELS and set(case.get('accept', [])) <= LABELS - {case['intent']}, case['id']
        assert case['critical'] in (True, False, None) and case['split'] in ('dev', 'test'), case['id']
        assert set(case['entities']) <= {'order_id', 'amount', 'error_code', 'date'}, case['id']
        for option in case['agents'] or []:
            assert option and set(option) <= AGENTS, case['id']
        for values in case['entities'].values():
            assert all(v in case['message'] for v in values), case['id']
        if case['critical'] is True:
            assert case['agents'] == [['escalation']], case['id']
    for split in ('dev', 'test'):
        assert {c['intent'] for c in INTENTS if c['split'] == split} == LABELS


def test_intent_cases_do_not_leak_from_prompt_templates_or_default_suite():
    known = [t for texts in _TEMPLATES.values() for t in texts] + [c.message for c in DEFAULT_INTENT_CASES]
    for case in INTENTS:
        if case.get('history'):
            continue        # 多轮用例的短句要结合历史判断
        mine = grams(case['message'])
        for text in known:
            overlap = len(mine & grams(text)) / len(mine | grams(text))
            assert overlap < 0.5, (case['id'], case['message'], text)


def test_retrieval_answers_exist_in_relevant_documents():
    docs = {d['id']: d for d in CORPUS}
    assert len(docs) == len(CORPUS) == len({d['title'] for d in CORPUS})
    assert len({q['id'] for q in QUERIES}) == len(QUERIES) >= 50
    for case in QUERIES:
        assert set(case['relevant']) <= set(docs), case['id']
        assert bool(case['relevant']) == bool(case['answer']) == ('unanswerable' not in case['tags']), case['id']
        for span in case['answer']:
            assert any(span in docs[d]['content'] for d in case['relevant']), (case['id'], span)


def test_english_queries_mirror_chinese_cases():
    base = {q['id']: q for q in QUERIES}
    english = bench.load_jsonl('retrieval_en.jsonl')
    assert len({q['id'] for q in english}) == len(english) >= 50
    for case in english:
        origin = base[case['id'].removeprefix('en-')]
        assert case['query'].isascii() and case['tags'][0] == 'english', case['id']
        assert all(case[key] == origin[key] for key in ('relevant', 'answer', 'split')), case['id']


def test_corpus_default_documents_match_knowledge_base_seed():
    from tooling.knowledge_base import KnowledgeBase
    seeded = []
    KnowledgeBase._load_default_docs(SimpleNamespace(add_documents=seeded.extend))
    expected = {d['title']: d['content'] for d in seeded}
    assert {d['title']: d['content'] for d in CORPUS if d['source'] == 'default'} == expected


def test_limited_selection_covers_every_label_before_repeating():
    picked = bench.select(INTENTS, limit=len(LABELS), group=lambda c: c['intent'])
    assert Counter(c['intent'] for c in picked) == Counter(LABELS)
    assert bench.select(INTENTS, 'dev') == [c for c in INTENTS if c['split'] == 'dev']


# ── 评分器：已知正确与已知错误 ───────────────────────────────────────────────

def test_oracle_scores_full_marks_and_majority_baseline_does_not():
    oracle = bench.summarize_intent(asyncio.run(bench.run_intent(INTENTS, 'oracle')))
    for key in ('accuracy', 'accuracy_lenient', 'route', 'route_given_gold', 'entities_exact'):
        assert oracle[key]['rate'] == 1.0, key
    assert oracle['critical']['precision'] == oracle['critical']['recall'] == 1.0
    majority = bench.summarize_intent(asyncio.run(bench.run_intent(INTENTS, 'majority')))
    assert majority['accuracy']['rate'] < 0.15 and majority['macro_f1'] < 0.1
    assert majority['critical']['recall'] == 0.0


def test_retrieval_oracle_and_empty_results_hit_both_ends():
    docs = {d['id']: d for d in CORPUS}

    async def oracle(case, top_k):
        return [{**docs[i], 'score': 1.0} for i in case['relevant']], None

    async def null(case, top_k):
        return [], None

    best = bench.summarize_retrieval(asyncio.run(bench.run_retrieval(QUERIES, CORPUS, oracle)))
    worst = bench.summarize_retrieval(asyncio.run(bench.run_retrieval(QUERIES, CORPUS, null)))
    for key in ('hit1', 'hit5', 'answer5'):
        assert best[key]['rate'] == 1.0 and worst[key]['rate'] == 0.0
    assert best['unanswerable']['returned_any'] == 0


def test_answer_requires_the_evidence_text_not_only_the_right_document():
    case = next(q for q in QUERIES if q['relevant'])
    title = next(d['title'] for d in CORPUS if d['id'] == case['relevant'][0])
    titles = {d['title']: d['id'] for d in CORPUS}
    row = bench.score_retrieval(case, [{'title': title, 'content': '同一文档的另一段', 'score': 0.5}], titles)
    assert row['hit1'] and not row['answer5']


def test_entity_scoring_normalizes_spacing_and_ignores_unlisted_dates():
    case = {'entities': {'amount': ['500 元']}}
    assert bench.entity_diff(case, {'amount': ['500元'], 'date': ['今天']}) == {
        'order_id': (0, 0, 0), 'amount': (1, 0, 0), 'error_code': (0, 0, 0)}
    assert bench.entity_diff(case, {'amount': [], 'error_code': ['500']})['error_code'] == (0, 1, 0)


def test_unassigned_routes_are_not_scored_and_order_does_not_matter():
    assert bench.route_ok({'agents': None}, ['billing']) is None
    assert bench.route_ok({'agents': [['technical', 'billing']]}, ['billing', 'technical']) is True
    assert bench.route_ok({'agents': [['billing']]}, ['billing', 'technical']) is False


# ── 失败不能伪装成模型结果 ────────────────────────────────────────────────────

def make_recognizer(create, monkeypatch):
    monkeypatch.setenv('INTENT_EMBEDDING_PROVIDER', 'local')
    recognizer = bench.BenchmarkRecognizer(api_key='test', model='test-model')
    recognizer.client = bench.SpyClient(SimpleNamespace(base_url='', messages=SimpleNamespace(create=create)))
    return recognizer


def live_rows(create, monkeypatch):
    return asyncio.run(bench.run_intent(INTENTS[:2], 'live', recognizer=make_recognizer(create, monkeypatch)))


def reply(text, stop='end_turn', model='test-model'):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=[SimpleNamespace(type='text', text=text)], stop_reason=stop, model=model, usage=usage)


def refund_reply(**kwargs):
    return reply('{"intent": "refund", "confidence": 0.9}', **kwargs)


def test_successful_live_call_records_usage_and_served_model(monkeypatch):
    async def create(**kwargs):
        return refund_reply()
    rows = live_rows(create, monkeypatch)
    assert [r['llm'] for r in rows] == ['ok', 'ok']
    assert all(r['llm_calls'] == 1 and r['input_tokens'] == 10 and r['served'] == ['test-model'] for r in rows)
    report = bench.build_report('intent', 'live', 'all', 1, 'd', rows, {'model': 'test-model'})
    assert report['status'] == 'complete'


@pytest.mark.parametrize('stop,expected', [('max_tokens', 'truncated'), ('refusal', 'refusal'), ('end_turn', 'unparseable')])
def test_missing_llm_answer_is_classified_by_cause(monkeypatch, stop, expected):
    async def create(**kwargs):
        return reply('', stop)
    rows = live_rows(create, monkeypatch)
    assert {r['llm'] for r in rows} == {expected}
    assert bench.summarize_intent(rows)['accuracy_llm_ok']['n'] == 0


def test_api_failure_and_timeout_make_the_report_partial(monkeypatch):
    class Overloaded(Exception):
        status_code = 529

    async def overloaded(**kwargs):
        raise Overloaded()

    async def slow(**kwargs):
        await asyncio.sleep(5)

    rows = live_rows(overloaded, monkeypatch)
    assert {r['llm'] for r in rows} == {'infra:Overloaded:529'}
    assert bench.build_report('intent', 'live', 'all', 1, 'd', rows, {'model': 'test-model'})['status'] == 'partial'
    monkeypatch.setenv('KORA_INTENT_TIMEOUT_S', '0.1')
    assert {r['llm'] for r in live_rows(slow, monkeypatch)} == {'infra:timeout'}


def test_served_model_mismatch_is_reported(monkeypatch):
    async def create(**kwargs):
        return refund_reply(model='other-model')
    report = bench.build_report('intent', 'live', 'all', 1, 'd', live_rows(create, monkeypatch), {'model': 'test-model'})
    assert report['model_mismatches'] == ['other-model'] and report['status'] == 'partial'


def test_repeated_runs_do_not_hit_the_intent_cache(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(1)
        return refund_reply()
    recognizer = make_recognizer(create, monkeypatch)
    asyncio.run(bench.run_intent(INTENTS[:2], 'live', reps=3, recognizer=recognizer))
    assert len(calls) == 6


def test_retrieval_error_is_not_scored_as_a_miss():
    async def broken(case, top_k):
        return [], '检索失败'
    rows = asyncio.run(bench.run_retrieval(QUERIES[:3], CORPUS, broken))
    assert bench.summarize_retrieval(rows)['hit5']['n'] == 0
    assert bench.report_status('retrieval', rows, []) == 'partial'


def test_e2e_records_timeout_and_errors_without_losing_other_cases():
    class Orchestrator:
        _intent_recognizer = SimpleNamespace(_cache={})

        async def run(self, req):
            if req.conv_id.startswith('a'):
                await asyncio.sleep(5)
            if req.conv_id.startswith('b'):
                raise RuntimeError('boom')
            return SimpleNamespace(status='success', agent_types=[], tools_used=['search_knowledge_base'], response='ok')

    cases = [{'id': i, 'message': 'x', 'split': 'dev', 'tags': []} for i in 'abc']
    rows = asyncio.run(bench.run_e2e(cases, Orchestrator(), concurrency=3, timeout=0.1))
    assert [r['status'] for r in rows] == ['timeout', 'error:RuntimeError', 'success']
    assert bench.summarize_e2e(rows)['tool_calls'] == {'search_knowledge_base': 1}
    assert bench.report_status('e2e', rows, []) == 'partial'


# ── 统计与对比 ────────────────────────────────────────────────────────────────

def test_statistics_helpers():
    assert bench.wilson(0, 0) is None
    low, high = bench.wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    assert [bench.pct([4, 1, 3, 2], q) for q in (50, 95)] == [2, 4] and bench.pct([], 50) is None
    assert bench.mcnemar(0, 0) == 1.0 and bench.mcnemar(5, 5) == 1.0
    assert bench.mcnemar(10, 0) == pytest.approx(0.002, abs=1e-3)
    assert bench.auc([2, 3], [1]) == 1.0 and bench.auc([1], [1]) == 0.5 and bench.auc([], [1]) is None


def test_compare_pairs_cases_and_rejects_a_changed_dataset(tmp_path):
    def report(flags, dataset='d'):
        return {'suite': 'retrieval', 'dataset': dataset, 'status': 'complete',
                'rows': [{'id': str(i), 'hit1': f, 'hit5': True} for i, f in enumerate(flags)]}
    before, after = report([False] * 8 + [True]), report([True] * 8 + [False])
    result = bench.compare(before, after)['hit1']
    assert result['fixed'] == [str(i) for i in range(8)] and result['broken'] == ['8'] and result['p'] < 0.05
    with pytest.raises(ValueError, match='dataset'):
        bench.compare(before, report([True] * 9, dataset='other'))
    path = tmp_path / 'r.json'
    bench.save_report(after, path)
    assert json.loads(path.read_text(encoding='utf-8')) == after


def test_fingerprint_changes_with_content_only():
    assert bench.fingerprint(INTENTS) == bench.fingerprint([dict(c) for c in INTENTS])
    assert bench.fingerprint(INTENTS) != bench.fingerprint(INTENTS[:-1])


def test_evaluation_kb_never_uses_a_chroma_server(monkeypatch):
    import tooling.knowledge_base as module

    class Connected:
        _use_server, doc_count = True, 0

        def __init__(self, **kwargs):
            assert os.environ['KORA_ENV'] == 'development'

    monkeypatch.setattr(module, 'KnowledgeBase', Connected)
    monkeypatch.setenv('KORA_ENV', 'production')
    with pytest.raises(RuntimeError, match='临时库'):
        with bench.evaluation_kb(CORPUS):
            pass
    assert os.environ['KORA_ENV'] == 'production'
