import json
from pathlib import Path
import random

import pytest

from prism import scoring_protocol as protocol
from prism.io import digest, dump, receipt, write_rows


def test_fixed_checker_rng_repeats_both_generators_and_restores_state():
    langdetect = pytest.importorskip('langdetect')
    old_seed = langdetect.DetectorFactory.seed
    state = random.getstate()
    def draw():
        return [random.randint(0, 10000) for _ in range(8)], langdetect.detect_langs('Otec matka syn.')
    with protocol.fixed_checker_rng():
        first = draw()
    with protocol.fixed_checker_rng():
        second = draw()
    assert first[0] == second[0]
    assert [(x.lang,x.prob) for x in first[1]] == [(x.lang,x.prob) for x in second[1]]
    assert random.getstate() == state and langdetect.DetectorFactory.seed == old_seed
    with pytest.raises(RuntimeError):
        with protocol.fixed_checker_rng():
            draw()
            raise RuntimeError('interrupted checker')
    assert random.getstate() == state and langdetect.DetectorFactory.seed == old_seed


def fixture_score(value):
    record={'instruction_id_list':['keywords:letter_frequency'], 'follow_instruction_list':[value]}
    return {'metrics':{'prompt_level_strict':float(value)}, 'score':float(value),
            'details':[{'id':'one','key':1,'strict':record,'loose':record}]}


def test_sidecar_preserves_original_and_rejects_modified_generated_input(tmp_path,monkeypatch):
    pytest.importorskip('langdetect')
    directory=tmp_path/'reference';directory.mkdir()
    original=directory/'benchmark.metrics.json';dump(original,fixture_score(False))
    responses=directory/'benchmark.responses.jsonl';write_rows(responses,[{'id':'one','response':'letters'}])
    marker=directory/'reference.receipt.json';receipt(marker,[original,responses])
    frozen={p:digest(p) for p in (original,responses,marker)}
    c={'choices':{'benchmark':{'kind':'ifeval','official_source_revision':'f'*40}}}
    monkeypatch.setattr(protocol,'score_benchmark',lambda *args:fixture_score(True))
    result=protocol.materialize(c,directory)
    assert result['score']==1 and len(result['scoring_protocol']['changed_flags'])==2
    assert all(digest(p)==sha for p,sha in frozen.items())
    assert protocol.materialize(c,directory)==result
    responses.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError):protocol.effective_benchmark(c,directory,required=True)


def test_unstable_fixed_rng_score_is_not_published(tmp_path,monkeypatch):
    pytest.importorskip('langdetect')
    directory=tmp_path/'reference';directory.mkdir()
    original=directory/'benchmark.metrics.json';dump(original,fixture_score(False))
    responses=directory/'benchmark.responses.jsonl';write_rows(responses,[{'id':'one','response':'text'}])
    receipt(directory/'reference.receipt.json',[original,responses])
    values=iter([fixture_score(False),fixture_score(True)])
    monkeypatch.setattr(protocol,'score_benchmark',lambda *args:next(values))
    c={'choices':{'benchmark':{'kind':'ifeval','official_source_revision':'f'*40}}}
    with pytest.raises(RuntimeError,match='not reproducible'):protocol.materialize(c,directory)
    assert not (directory/protocol.METRICS).exists() and not (directory/protocol.RECEIPT).exists()


def test_explicit_ifeval_scoring_rejects_missing_certification(tmp_path):
    dump(tmp_path/'benchmark.metrics.json',{'score':.5})
    c={'choices':{'benchmark':{'kind':'ifeval'}}}
    with pytest.raises(RuntimeError,match='requires certified'):protocol.effective_benchmark(c,tmp_path,required=True)
    assert protocol.effective_benchmark(c,tmp_path)=={'score':.5}
