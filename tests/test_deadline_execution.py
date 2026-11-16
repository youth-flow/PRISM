from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from prism import execution_authorization as delivery
from prism.api_budget import BudgetTransport,USDBudget
from prism.budgeted_judge import prepare_manifest,run_manifest,recover_manifest
from prism.io import digest,rows
from prism.local_judge import _load_packet,_records
from test_budgeted_judge import response
from test_evaluation_protocol import profiles
from test_evaluation_v2 import config,save_generations


def test_deadline_ceiling_revision_keeps_old_contract_dispatches_and_unknown_fees(tmp_path):
    budget=USDBudget(tmp_path/'budget.sqlite',total_usd=54,purposes={'calibration':3,'training':46,'evaluation':5})
    payload={'model':'gpt-6.1-sol','max_output_tokens':4096,'service_tier':'flex','input':[]}
    budget.reserve('old-unknown','training','gpt-6.1-sol',payload);budget.uncertain('old-unknown')
    old=budget.snapshot();budget.control(True,'New explicit user deadline/cap');budget.control_purpose('training',True,'Labels finished')
    with budget.transaction() as db:before=db.execute('SELECT * FROM dispatch').fetchall()
    with pytest.raises(ValueError):budget.authorize_limits(total_usd=63,purposes={'calibration':2,'training':31,'evaluation':30},reason='No authority artifact')
    budget.authorize_limits(total_usd=63,purposes={'calibration':2,'training':31,'evaluation':30},reason='Explicit later user USD30 evaluation cap',execution_authorization=delivery.canonical_path())
    assert budget.state()['limits_usd']['evaluation']==30 and budget.state()['limits_usd']['total']==63
    assert budget.limits['total']==54_000_000 and budget.snapshot()==old
    with budget.transaction() as db:assert db.execute('SELECT * FROM dispatch').fetchall()==before
    with pytest.raises(ValueError,match='match the explicit'):budget.authorize_limits(total_usd=100,purposes={'calibration':2,'training':31,'evaluation':67},reason='Too much',execution_authorization=delivery.canonical_path())


def test_standard_delivery_changes_only_tier_and_binds_actual_wire_and_cache(tmp_path):
    budget=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'evaluation':1})
    payload={'model':'gpt-6.1-sol','max_output_tokens':4096,'service_tier':'flex','input':[]};sent=[]
    def fake(url,wire,key,timeout):
        sent.append(wire);result,rid=response();result['service_tier']='default';return result,rid
    bound=BudgetTransport(budget,'evaluation',fake,stream_responses=True,standard_processing=True).for_call('one')
    value=bound('https://api.openai.com/v1/responses',payload,'TEST',900)
    assert sent==[{**payload,'stream':True,'service_tier':'default'}] and payload['service_tier']=='flex'
    assert bound.receipt['request_transport']==delivery.wire_evidence(payload,standard=True)[1]
    replay=BudgetTransport(budget,'evaluation',lambda *a:pytest.fail('No duplicate request')).for_call('one')
    assert replay('https://api.openai.com/v1/responses',payload,'TEST',900)==value
    assert replay.receipt['request_transport']==bound.receipt['request_transport'] and len(sent)==1
    assert budget.snapshot()['evaluation']['charged_usd_estimate']==28/1e6
    with pytest.raises(ValueError):BudgetTransport(budget,'training',standard_processing=True,stream_responses=True)


def test_explicit_standard_manifest_preserves_packets_successes_and_import_validation(tmp_path,monkeypatch):
    from prism.evaluation_protocol import export_bundle
    c=config(tmp_path);save_generations(c);primary,secondary=profiles(c,tmp_path)
    profile=yaml.safe_load(primary.read_text('utf-8'));profile['spending_usd']={'user_total':40,'operational_total':39,'calibration':2,'training':35.5,'evaluation':1.8}
    primary.write_text(yaml.safe_dump(profile),encoding='utf-8');export_bundle(c,'prism',8,11,primary,secondary)
    packet=Path(c['output'])/'prism/n8/seed11/evaluation.packet.json';before=digest(packet)
    grant=delivery.authorization();grant.update(original_experiment_sha256=profile['study_authorization']['original_experiment_sha256'],active_budgets=[8])
    monkeypatch.setattr(delivery,'authorization',lambda path=None:grant)
    budget=USDBudget(tmp_path/'budget.sqlite',total_usd=39,purposes={'calibration':2,'training':31,'evaluation':1.8});budget.control_purpose('training',True,'No labels')
    manifest=tmp_path/'standard.json';prepare_manifest([packet],primary,budget.path,manifest,workers=1,execution_authorization=delivery.canonical_path())
    monkeypatch.setenv('OPENAI_API_KEY','TEST');sent=[]
    # A real earlier Flex success remains authoritative after the delivery revision.
    from prism.judges import APIJudge
    original=_load_packet(packet);task=original['tasks'][0];call=task['calls'][0]
    prior=BudgetTransport(budget,'evaluation',response,stream_responses=True).for_call(f"{original['packet_id']}/{task['id']}/0/0")
    judge=APIJudge({'execution':{'allow_paid_api_calls':True},'judges':{'evaluation':original['settings']}},'evaluation',transport=prior)
    value=judge.call(task['messages'],task['left'],task['right'],call['seed'],call['flip']);value['budget_transport']=prior.receipt
    packet.with_name('evaluation.results.jsonl').write_text(json.dumps({'packet_id':original['packet_id'],'task_id':task['id'],'attempt':0,'result':value})+'\n',encoding='utf-8')
    def fake(url,wire,key,timeout):
        assert wire['model']=='gpt-6.1-sol' and wire['reasoning']['effort']=='high' and wire['service_tier']=='default' and wire['stream'] is True
        sent.append(wire);result,rid=response();result['service_tier']='default';return result,rid
    state=run_manifest(manifest,execute=True,transport=fake,stream_responses=True,poll_seconds=.01)
    assert state['state']=='completed' and state['responses_transport']=='sse_standard_v1' and len(sent)==7
    assert digest(packet)==before and _records(_load_packet(packet),packet.with_name('evaluation.results.jsonl'),with_accounting=True)[2]['actual_judge_calls']==8
    run_manifest(manifest,execute=True,transport=lambda *a:pytest.fail('Successes must not repeat'),stream_responses=True)
    budget.control(True,'Offline validation');assert recover_manifest(manifest)['recovered_records']==0
    data=rows(packet.with_name('evaluation.results.jsonl'));data[1]['result']['budget_transport']['request_transport']['execution_authorization_sha256']='0'*64
    bad=tmp_path/'bad.jsonl';bad.write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
    with pytest.raises(ValueError,match='streaming request payload'):_records(_load_packet(packet),bad)
