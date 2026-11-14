"""Raw requests remain immutable; aliases and transport failures replay explicitly."""
import copy
import json
from urllib.error import URLError

import pytest

from prism.evaluation_v2 import _training_accounting
from prism.io import digest, rows, verify_receipt, write_rows
from prism.judges import APIJudge
from prism.local_judge import export_packet, import_results, run_packet
from test_evaluation_v2 import config
from test_judges import preference_choice
from test_local_judge import fake_transport, training_fixture


def test_old_alias_is_replayed_without_replacing_or_repeating_any_raw_request(tmp_path, monkeypatch):
    c = config(tmp_path)
    out = training_fixture(c)
    packet, results = tmp_path/'packet.json', tmp_path/'results.jsonl'
    export_packet(c, 'label', 'prism', 8, 11, packet)
    data = json.loads(packet.read_text())
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'LOCAL_TEST_KEY')
    allowed = copy.deepcopy(c)
    allowed['execution']['allow_paid_api_calls'] = True
    generated = []
    for i, task in enumerate(data['tasks']):
        for attempt in (range(2) if i == 0 else range(1)):
            call = task['calls'][attempt]
            # An old decoder rejected the short alias; its later retry preferred
            # the opposite candidate. The first interpretable observation wins.
            text = json.dumps({'preferred_response': ('A' if call['flip'] else 'B') if i == 0 and attempt == 0
                               else ('response_B' if call['flip'] else 'response_A')})
            judge = APIJudge(allowed, transport=lambda *a: ({'choices':[preference_choice(text)]}, None))
            result = judge.call(task['messages'], task['left'], task['right'], call['seed'], call['flip'])
            if i == 0 and attempt == 0:
                assert result['label'] == 0
                result.update(label=None, error={'type':'invalid_judgment_format'})
            generated.append({'packet_id':data['packet_id'], 'task_id':task['id'], 'attempt':attempt, 'result':result})
    write_rows(results, generated)
    before = results.read_bytes()
    run_packet(packet, results, execute=True, transport=lambda *a: pytest.fail('No repeated API spending'))
    metadata = import_results(c, packet, results)
    assert results.read_bytes() == before
    labels = rows(out/'labels.jsonl')
    assert labels[0]['label'] == 0
    assert labels[0]['output_normalization']['source_decoded_label'] is None
    assert labels[0]['output_normalization']['source_error']['type'] == 'invalid_judgment_format'
    assert metadata['valid_labels'] == 8 and metadata['actual_judge_calls'] == 9
    assert metadata['canonical_judge_calls'] == 8 and metadata['superseded_semantic_calls'] == 1
    assert metadata['source_invalid_calls'] == 1 and metadata['invalid_calls'] == 0
    assert len(rows(out/'judge_calls.jsonl')) == 8
    assert _training_accounting(c)['unique_training_calls'] == 9
    assert _training_accounting(c)['unique_valid_observations'] == 8
    verify_receipt(out/'labels.receipt.json')


def test_network_retry_reuses_exhausted_final_request_and_preserves_partial_history(tmp_path, monkeypatch):
    c = config(tmp_path)
    original = copy.deepcopy(c)
    out = training_fixture(c)
    packet, results = tmp_path/'packet.json', tmp_path/'results.jsonl'
    export_packet(c, 'label', 'prism', 8, 11, packet)
    packet_hash = digest(packet)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'LOCAL_TEST_KEY')
    seen = []
    def flaky(*args):
        seen.append(copy.deepcopy(args[1]))
        if len(seen) <= 3:
            raise URLError('TEST temporary connection loss')
        return fake_transport(*args)
    with pytest.raises(RuntimeError, match='exhausted attempts'):
        run_packet(packet, results, execute=True, transport=flaky)
    before = results.read_bytes()
    assert len(seen) == 3
    with pytest.raises(RuntimeError):
        import_results(c, packet, results)
    partial = (out/'judge_calls.jsonl').read_bytes()
    assert len(rows(out/'judge_calls.jsonl')) == 3
    with pytest.raises(RuntimeError, match='exhausted attempts'):
        run_packet(packet, results, execute=True, transport=lambda *a: pytest.fail('Default remains three requests'))
    run_packet(packet, results, execute=True, transport=flaky, network_retries=2)
    assert seen[2] == seen[3]
    assert results.read_bytes().startswith(before)
    assert digest(packet) == packet_hash and c == original
    all_rows = rows(results)
    assert len(all_rows) == 11
    assert all_rows[3]['attempt'] == 2 and all_rows[3]['transport_retry'] == 1
    assert all_rows[2]['result']['seed'] == all_rows[3]['result']['seed']
    assert all_rows[2]['result']['flip'] == all_rows[3]['result']['flip']
    metadata = import_results(c, packet, results)
    assert metadata['actual_judge_calls'] == 11 and metadata['valid_labels'] == 8
    assert metadata['canonical_judge_calls'] == 10 and metadata['collapsed_transport_calls'] == 1
    assert metadata['max_transport_retry_used'] == 1
    archives = list((out/'label_derivations').glob('*/judge_calls.jsonl'))
    assert len(archives) == 1 and archives[0].read_bytes() == partial
    assert len(rows(out/'labels.jsonl')) == 8
    verify_receipt(archives[0].parent/'receipt.json')
    assert _training_accounting(c)['unique_training_calls'] == 11


def test_extra_transport_retries_never_extend_semantic_format_retry_budget(tmp_path, monkeypatch):
    c = config(tmp_path)
    training_fixture(c)
    packet, results = tmp_path/'packet.json', tmp_path/'results.jsonl'
    export_packet(c, 'label', 'prism', 8, 11, packet)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'LOCAL_TEST_KEY')
    seen = []
    def malformed(*args):
        seen.append(True)
        return {'choices':[preference_choice('{"preferred_response":"D"}')]}, None
    with pytest.raises(RuntimeError, match='exhausted attempts'):
        run_packet(packet, results, execute=True, transport=malformed, network_retries=2)
    assert len(seen) == len(rows(results)) == 3


def test_retransmission_is_bounded_and_cannot_advance_to_a_new_seed(tmp_path, monkeypatch):
    c = config(tmp_path)
    training_fixture(c)
    packet, results = tmp_path/'packet.json', tmp_path/'results.jsonl'
    export_packet(c, 'label', 'prism', 8, 11, packet)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'LOCAL_TEST_KEY')
    def network(*args):
        raise URLError('TEST connection loss')
    with pytest.raises(RuntimeError, match='exhausted attempts'):
        run_packet(packet, results, execute=True, transport=network, network_retries=2)
    data = rows(results)
    assert len(data) == 5
    assert [r['attempt'] for r in data] == [0, 1, 2, 2, 2]
    assert data[-1]['result']['seed'] == data[-3]['result']['seed']
    with pytest.raises(RuntimeError, match='exhausted attempts'):
        run_packet(packet, results, execute=True, transport=lambda *a: pytest.fail('No unbounded retry'), network_retries=2)
