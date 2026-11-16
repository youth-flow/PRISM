from copy import deepcopy
import io,json,time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
import yaml

from prism import execution_authorization as delivery,budgeted_judge
from prism.api_budget import USDBudget,BudgetTransport
from prism.judges import APIJudge,FlexCapacityUnavailable,_read_response_events
from prism.io import rows
from prism.local_judge import _load_packet,_records,import_results
from prism.budgeted_judge import prepare_manifest,run_manifest,recover_manifest
from prism.evaluation_protocol import export_bundle
from test_evaluation_protocol import profiles
from test_evaluation_v2 import config,save_generations
from test_budgeted_judge import response

ENCODED=json.dumps({'type':'error','error':{'type':'resource_unavailable','code':'flex_unavailable','message':'Unavailable','param':None,'headers':{'retry-after':'300'}}}).encode()

def prepared(tmp_path,monkeypatch):
 c=config(tmp_path);save_generations(c);primary,secondary=profiles(c,tmp_path)
 profile=yaml.safe_load(primary.read_text('utf-8'));profile['spending_usd']={'user_total':40,'operational_total':39,'calibration':2,'training':31,'evaluation':1.8};primary.write_text(yaml.safe_dump(profile),encoding='utf-8')
 export_bundle(c,'prism',8,11,primary,secondary);packet=Path(c['output'])/'prism/n8/seed11/evaluation.packet.json'
 grant=delivery.authorization();grant.update(original_experiment_sha256=profile['study_authorization']['original_experiment_sha256'],active_budgets=[8]);monkeypatch.setattr(delivery,'authorization',lambda path=None:grant)
 budget=USDBudget(tmp_path/'budget.sqlite',total_usd=39,purposes={'calibration':2,'training':31,'evaluation':1.8});budget.control_purpose('training',True,'Complete')
 manifest=tmp_path/'manifest.json';prepare_manifest([packet],primary,budget.path,manifest,workers=1,execution_authorization=delivery.canonical_path(),service_tier='flex',fallback_to_standard=True)
 monkeypatch.setenv('OPENAI_API_KEY','TEST');monkeypatch.setattr(budgeted_judge,'time',SimpleNamespace(monotonic=time.monotonic,sleep=lambda _:None))
 return c,packet,budget,manifest

@pytest.mark.parametrize('deny',[False,True])
def test_same_judge_fallback_uses_only_original_slots_and_accounts_every_request(tmp_path,monkeypatch,deny):
 c,path,budget,manifest=prepared(tmp_path,monkeypatch);sent=[]
 def fake(url,wire,key,timeout):
  sent.append(deepcopy(wire));assert wire['model']=='gpt-6.1-sol' and wire['reasoning']['effort']=='high' and timeout==900 and wire['stream'] is True
  if deny and wire['service_tier']=='flex':raise FlexCapacityUnavailable(ENCODED)
  value,rid=response();value['service_tier']=wire['service_tier'];return value,rid
 result=run_manifest(manifest,execute=True,transport=fake,stream_responses=True,poll_seconds=.01)
 assert result['state']=='completed' and len(sent)==(16 if deny else 8)
 assert len(result['capacity_refusals'])==(8 if deny else 0) and not result['transport_failures']
 raw=path.with_name('evaluation.results.jsonl');events=rows(raw);packet=_load_packet(path)
 assert all(event['attempt'] in ([0,1] if deny else [0]) for event in events)
 assert _records(packet,raw,with_accounting=True)[2]['actual_judge_calls']==len(sent)
 assert budget.snapshot()['evaluation']['unresolved_reservations_usd']==0
 with budget.transaction() as db:assert db.execute("SELECT COUNT(*) FROM dispatch WHERE status='failed_no_charge' AND charged=0").fetchone()[0]==(8 if deny else 0)
 assert import_results(c,path,raw)['evaluation_judge_calls']==len(sent)
 run_manifest(manifest,execute=True,transport=lambda *a:pytest.fail('No repeated success'),stream_responses=True)

def test_capacity_response_before_result_append_recovers_without_resending(tmp_path,monkeypatch):
 c,path,budget,manifest=prepared(tmp_path,monkeypatch);packet=_load_packet(path);task=packet['tasks'][0];call=task['calls'][0]
 def denied(*args):raise FlexCapacityUnavailable(ENCODED)
 bound=BudgetTransport(budget,'evaluation',denied,stream_responses=True).for_call(f"{packet['packet_id']}/{task['id']}/0/0")
 judge=APIJudge({'execution':{'allow_paid_api_calls':True},'judges':{'evaluation':packet['settings']}},'evaluation',transport=bound)
 value=judge.call(task['messages'],task['left'],task['right'],call['seed'],call['flip']);assert value['error']['type']=='capacity_unavailable'
 budget.control(True,'Crash after received refusal')
 assert recover_manifest(manifest)['recovered_records']==1
 event=rows(path.with_name('evaluation.results.jsonl'))[0]
 assert event['attempt']==0 and event['result']['capacity_refusal']==value['capacity_refusal']
 assert budget.snapshot()['evaluation']['dispatches']==1 and budget.snapshot()['evaluation']['charged_usd_estimate']==0

def test_stream_capacity_classification_rejects_unknown_or_ambiguous_errors(tmp_path):
 with pytest.raises(FlexCapacityUnavailable):_read_response_events(io.BytesIO(b'data: '+ENCODED+b'\n\n'))
 for data in [b'{"type":"error","type":"other"}',json.dumps({'type':'error','error':{'type':'resource_unavailable','code':'other'}}).encode(),json.dumps({'type':'error','error':{'type':'resource_unavailable','code':'flex_unavailable'},'usage':{'output_tokens':2}}).encode()]:
  with pytest.raises(ValueError):FlexCapacityUnavailable(data)

def test_fallback_never_expands_the_three_request_limit(tmp_path,monkeypatch):
 c,path,budget,manifest=prepared(tmp_path,monkeypatch);sent=[]
 def fail(url,wire,key,timeout):
  sent.append(wire)
  if wire['service_tier']=='flex':raise FlexCapacityUnavailable(ENCODED)
  raise URLError('Network unavailable')
 with pytest.raises(RuntimeError):run_manifest(manifest,execute=True,transport=fail,stream_responses=True,poll_seconds=.01)
 assert len(sent)==3 and [x['service_tier'] for x in sent]==['flex','default','default']
 assert [e['attempt'] for e in rows(path.with_name('evaluation.results.jsonl'))]==[0,1,2]
