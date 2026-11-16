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


@pytest.mark.parametrize('failure',[HTTPError('https://api.openai.com/v1/responses',429,'Flex capacity',{},None),HTTPError('https://api.openai.com/v1/responses',503,'Temporary unavailable',{},None)])
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


def test_network_outage_trips_shared_breaker_and_preserves_resume(tmp_path,monkeypatch):
    manifest,b,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    sent=[]
    def outage(*args):
        sent.append(True);raise URLError('proxy disconnected')
    with pytest.raises(DispatchPaused,match='stopped'):
        run_manifest(manifest,execute=True,transport=outage,poll_seconds=.01)
    prefix=rows(tmp_path/'train.results.jsonl')
    assert 1<=len(sent)<=4 and len(prefix)==len(sent)
    assert len(sent)>=2 and all(r['result']['label'] is None for r in prefix)
    assert all(r['attempt']<=1 for r in prefix)
    b.control(False,'Network restored; checked resume')
    result=run_manifest(manifest,execute=True,transport=response,poll_seconds=.01)
    assert result['state']=='completed'
    assert rows(tmp_path/'train.results.jsonl')[:len(prefix)]==prefix
    assert b.snapshot()['training']['dispatches']==8+len(sent)
    assert b.snapshot()['training']['unresolved_reservations_usd']>0


def test_isolated_packet_loss_keeps_original_semantic_slots_and_does_not_stop_batch(tmp_path,monkeypatch):
    manifest,b,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    sent=[]
    def transient(*args):
        sent.append(True)
        if len(sent)==1:raise URLError('one isolated packet loss')
        return response(*args)
    result=run_manifest(manifest,execute=True,transport=transient,poll_seconds=.01)
    assert result['state']=='completed' and len(sent)==9
    assert result['network_breaker']=={'errors':2,'window_seconds':60}
    assert not b.state()['paused'] and b.snapshot()['training']['unresolved_reservations_usd']>0
    assert result['transport_failures'][0]['exception_type']=='URLError'
    failed=next(r for r in rows(tmp_path/'train.results.jsonl') if r['result']['label'] is None)
    assert failed['result']['budget_transport']['transport_failure']['exception_type']=='URLError'


def test_transport_diagnostics_preserve_ssl_codes_without_exception_text():
    import ssl
    from prism.budgeted_judge import _transport_error_metadata
    error=ssl.SSLEOFError(8,'SECRET_KEY must never enter logs')
    value=_transport_error_metadata(URLError(error))
    assert value['exception_type']=='URLError' and value['reason_type']=='SSLEOFError' and value['errno']==8
    assert 'SECRET_KEY' not in json.dumps(value) and 'must never' not in json.dumps(value)


def test_bounded_network_supervision_resumes_without_new_slots_or_duplicate_successes(tmp_path,monkeypatch):
    from threading import Lock
    from prism.budgeted_judge import supervise_manifest
    manifest,b,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    lock=Lock();sent=[];checks=[];waits=[]
    def transient(*args):
        with lock:sent.append(True);fail=len(sent)<=2
        if fail:raise URLError('transient disconnect')
        return response(*args)
    def healthy():checks.append(True);return True
    result=supervise_manifest(manifest,execute=True,transport=transient,poll_seconds=.01,network_retries=2,
                             max_network_resumes=2,healthcheck=healthy,sleeper=waits.append)
    assert result['state']=='completed' and len(sent)==10 and checks==[True] and waits==[30]
    events=rows(tmp_path/'train.results.jsonl')
    assert len(events)==10 and len({r['task_id'] for r in events if r['result']['label'] is not None})==8
    assert not any(r.get('transport_retry') for r in events)
    assert b.snapshot()['training']['dispatches']==10 and b.snapshot()['training']['unresolved_reservations_usd']>0
    assert len(list((tmp_path/'network_pause_history').glob('*.json')))==1


def test_supervision_does_not_resume_user_pause_or_exhausted_formats(tmp_path,monkeypatch):
    from prism.budgeted_judge import supervise_manifest,_resume_own_network_pause
    manifest,b,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    b.control(True,'User pauses work')
    with pytest.raises(DispatchPaused):_resume_own_network_pause(b)
    with pytest.raises(DispatchPaused):
        supervise_manifest(manifest,execute=True,max_network_resumes=2,healthcheck=lambda:pytest.fail('No probe'),transport=lambda *args:pytest.fail('No HTTP'))
    assert b.state()['pause_reason']=='User pauses work'
    b.control(False,'Start fake formats');calls=[]
    def malformed(*args):
        calls.append(True);value,rid=response(*args);value['output'][0]['content'][0]['text']='C';return value,rid
    # A single worker makes the exact exhausted semantic prefix observable.
    data=json.loads(manifest.read_text('utf-8'));data['workers_per_packet']=1;manifest.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(RuntimeError,match='exhausted'):
        supervise_manifest(manifest,execute=True,max_network_resumes=2,healthcheck=lambda:pytest.fail('No format retry'),transport=malformed)
    assert len(calls)==3


def test_supervision_health_failure_keeps_all_costs_and_pause(tmp_path,monkeypatch):
    from prism.budgeted_judge import supervise_manifest
    manifest,b,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    def outage(*args):raise URLError('offline')
    with pytest.raises(DispatchPaused,match='health check'):
        supervise_manifest(manifest,execute=True,max_network_resumes=2,transport=outage,healthcheck=lambda:False,sleeper=lambda _:None)
    assert b.state()['paused'] and b.snapshot()['training']['unresolved_reservations_usd']>0


@pytest.mark.parametrize('failure',[{'exception_type':'BudgetExceeded'}, {'exception_type':'HTTPError','http_status':401}, {'exception_type':'ValueError'}])
def test_network_recovery_never_accepts_budget_auth_or_internal_failure(failure):
    import os
    from prism.budgeted_judge import _only_network_pause,_WORKER_PAUSE_REASON
    status={'state':'paused','error_type':'DispatchPaused','pid':os.getpid(),
            'transport_failures':[{'exception_type':'RemoteDisconnected'},failure]}
    assert not _only_network_pause(status,{'paused':True,'pause_reason':_WORKER_PAUSE_REASON})


def test_health_check_refuses_redirects_and_missing_credentials(monkeypatch):
    from prism import budgeted_judge as module
    monkeypatch.delenv('PRISM_HEALTH_KEY',raising=False)
    assert module._model_healthcheck({'api_key_env':'PRISM_HEALTH_KEY'}) is False
    monkeypatch.setenv('PRISM_HEALTH_KEY','TEST_KEY');calls=[]
    class Opener:
        def open(self,request,timeout):
            calls.append(request.full_url);assert timeout==15
            raise HTTPError(request.full_url,302,'redirect',{},None)
    def build(handler):
        assert isinstance(handler,module._NoRedirect);return Opener()
    monkeypatch.setattr(module,'build_opener',build)
    assert module._model_healthcheck({'api_key_env':'PRISM_HEALTH_KEY'}) is False
    assert calls==['https://api.openai.com/v1/models']


def test_cached_final_slot_retransmission_recovers_without_another_request(tmp_path,monkeypatch):
    from prism.local_judge import run_packet
    manifest,b,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    def outage(*args):raise URLError('lost response')
    with pytest.raises(RuntimeError,match='exhausted'):
        run_packet(path,tmp_path/'train.results.jsonl',execute=True,transport=BudgetTransport(b,'training',outage))
    packet=_load_packet(path);task=packet['tasks'][0];call=task['calls'][-1]
    bound=BudgetTransport(b,'training',response).for_call(f"{packet['packet_id']}/0/2/1")
    judge=APIJudge({'execution':{'allow_paid_api_calls':True},'judges':{'training':packet['settings']}},'training',transport=bound)
    assert judge.call(task['messages'],task['left'],task['right'],call['seed'],call['flip'])['label'] is not None
    before=b.snapshot();b.control(True,'Offline repair')
    result=recover_manifest(manifest)
    assert result['recovered_records']==1 and result['http_requests_sent']==0
    last=rows(tmp_path/'train.results.jsonl')[-1]
    assert last['attempt']==2 and last['transport_retry']==1 and last['result']['label'] is not None
    assert b.snapshot()==before
    assert recover_manifest(manifest)['recovered_records']==0


def test_cached_no_charge_error_recovers_as_failed_attempt_without_new_http(tmp_path,monkeypatch):
    import io
    manifest,b,path=prepared(tmp_path);packet=_load_packet(path);task=packet['tasks'][0];call=task['calls'][0]
    monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    def unavailable(*args):
        raise HTTPError(args[0],429,'Resource unavailable',{'x-request-id':'capacity'},io.BytesIO(b'{"error":{"code":"resource_unavailable"}}'))
    bound=BudgetTransport(b,'training',unavailable).for_call(f"{packet['packet_id']}/0/0/0")
    judge=APIJudge({'execution':{'allow_paid_api_calls':True},'judges':{'training':packet['settings']}},'training',transport=bound)
    assert judge.call(task['messages'],task['left'],task['right'],call['seed'],call['flip'])['label'] is None
    b.control(True,'Recover received capacity error before continuing')
    assert inspect_manifest(manifest)['recovery'][0]['cached_missing_jsonl']==1
    recovered=recover_manifest(manifest)
    assert recovered['http_requests_sent']==0 and recovered['recovered_records']==1
    result=rows(tmp_path/'train.results.jsonl')[0]['result']
    assert result['label'] is None and result['error']['status']==429
    assert result['budget_transport']['documented_no_charge'] is True
    assert b.snapshot()['training']['dispatches']==1 and b.snapshot()['training']['unresolved_reservations_usd']==0
