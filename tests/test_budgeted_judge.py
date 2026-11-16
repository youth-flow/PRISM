import json
from urllib.error import HTTPError,URLError

import pytest
import yaml

from prism.api_budget import BudgetTransport, DispatchPaused, USDBudget
from prism.budgeted_judge import inspect_manifest, owner_lock, prepare_manifest, recover_manifest, run_manifest
from prism.io import rows
from prism.judges import APIJudge
from prism.local_judge import _load_packet, export_packet
from test_evaluation_v2 import config
from test_local_judge import training_fixture


def prepared(tmp_path):
    c=config(tmp_path)
    c['judges']['training']={**c['judges']['evaluation'],'model':'gpt-6.1-sol','reasoning_effort':'high','service_tier':'flex','timeout_seconds':900}
    training_fixture(c)
    packet=tmp_path/'train.packet.json';export_packet(c,'label','prism',8,11,packet)
    money={'user_total':40,'operational_total':39,'calibration':2,'training':35.5,'evaluation':1.8}
    profile=tmp_path/'judges.yaml';profile.write_text(yaml.safe_dump({'judges':c['judges'],'spending_usd':money}),encoding='utf-8')
    ledger=tmp_path/'budget.sqlite';b=USDBudget(ledger,total_usd=39,purposes={k:money[k] for k in ('calibration','training','evaluation')})
    manifest=tmp_path/'run.json';prepare_manifest([packet],profile,ledger,manifest,workers=4)
    return manifest,b,packet


def response(*args):
    return {'status':'completed','id':'test-response','service_tier':'flex',
            'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'A'}]}],
            'usage':{'input_tokens':9,'output_tokens':1}},'test'


def test_manifest_requires_explicit_execution_and_unpaused_ledger(tmp_path,monkeypatch):
    manifest,b,packet=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    with pytest.raises(RuntimeError,match='--execute'):run_manifest(manifest,transport=lambda *args:pytest.fail('No HTTP'))
    b.control(True,'User requests code review')
    with pytest.raises(DispatchPaused):run_manifest(manifest,execute=True,transport=lambda *args:pytest.fail('No HTTP'))
    assert inspect_manifest(manifest)['http_requests_sent']==0
    b.control(False,'Explicit start for fake test')
    sent=[]
    def fake(*args):sent.append(True);return response(*args)
    result=run_manifest(manifest,execute=True,transport=fake,poll_seconds=.01)
    assert result['state']=='completed' and len(sent)==8
    run_manifest(manifest,execute=True,transport=lambda *args:pytest.fail('Must not repeat successful calls'))
    assert b.snapshot()['training']['dispatches']==8


def test_offline_recovery_after_received_response_missing_from_jsonl(tmp_path,monkeypatch):
    manifest,b,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    packet=_load_packet(path);task=packet['tasks'][0];call=task['calls'][0]
    identity=f"{packet['packet_id']}/0/0/0"
    bound=BudgetTransport(b,'training',response).for_call(identity)
    judge=APIJudge({'execution':{'allow_paid_api_calls':True},'judges':{'training':packet['settings']}},'training',transport=bound)
    assert judge.call(task['messages'],task['left'],task['right'],call['seed'],call['flip'])['label'] is not None
    b.control(True,'Recover cached response before restart')
    result=recover_manifest(manifest)
    assert result['recovered_records']==1 and result['http_requests_sent']==0
    recovered=rows(tmp_path/'train.results.jsonl')[0]
    assert recovered['result']['recovery']['offline'] and recovered['result']['budget_transport']['replayed_response']
    assert b.snapshot()['training']['dispatches']==1
    assert recover_manifest(manifest)['recovered_records']==0


def test_unknown_missing_response_is_recorded_without_inventing_label_or_spending(tmp_path):
    manifest,b,path=prepared(tmp_path);packet=_load_packet(path);task=packet['tasks'][0];call=task['calls'][0]
    judge=APIJudge({'judges':{'training':packet['settings']}},'training')
    from prism.judges import _messages
    left,right=(task['right'],task['left']) if call['flip'] else (task['left'],task['right'])
    payload=judge._payload(_messages(task['messages'],left,right,'training'))
    b.reserve(f"{packet['packet_id']}/0/0/0",'training','gpt-6.1-sol',payload)
    b.control(True,'Dead worker had an unknown response')
    before=b.snapshot()['training']['unresolved_reservations_usd']
    result=recover_manifest(manifest)
    assert result['http_requests_sent']==0 and result['recovered_records']==1
    value=rows(tmp_path/'train.results.jsonl')[0]['result']
    assert value['label'] is None and value['error']['type']=='response_unavailable'
    assert b.snapshot()['training']['unresolved_reservations_usd']==before


def test_ledger_owner_lock_prevents_a_second_worker(tmp_path):
    with owner_lock(tmp_path/'owner.lock'):
        with pytest.raises(RuntimeError,match='Another Judge worker'):
            with owner_lock(tmp_path/'owner.lock'):pytest.fail('Second owner')


@pytest.mark.parametrize('failure',[HTTPError('https://api.openai.com/v1/responses',429,'Flex capacity',{},None),URLError('temporary connection reset')])
def test_transient_transport_failure_uses_frozen_retry_without_global_abort(tmp_path,monkeypatch,failure):
    manifest,b,packet=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    from threading import Lock
    lock=Lock();calls=[]
    def transient(*args):
        with lock:
            calls.append(True);first=len(calls)==1
        if first:raise failure
        return response(*args)
    result=run_manifest(manifest,execute=True,transport=transient,poll_seconds=.01)
    records=rows(tmp_path/'train.results.jsonl')
    assert result['state']=='completed' and not result['ledger']['paused']
    assert len(calls)==9 and len(records)==9
    failed=[r for r in records if r['result']['label'] is None]
    assert len(failed)==1 and failed[0]['attempt']==0
    task=failed[0]['task_id']
    assert any(r['task_id']==task and r['attempt']==1 and r['result']['label'] is not None for r in records)
    assert b.snapshot()['training']['dispatches']==9
    assert b.snapshot()['training']['unresolved_reservations_usd']>0
    run_manifest(manifest,execute=True,transport=lambda *args:pytest.fail('Never repeat completed requests'))
