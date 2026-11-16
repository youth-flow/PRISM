import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from prism.api_budget import BudgetTransport,USDBudget
from prism.budgeted_judge import recover_manifest,run_manifest
from prism.io import rows
from prism.judges import _read_response_events
from prism.local_judge import _load_packet,_records
from test_budgeted_judge import prepared,response


def event(value):return ('data: '+json.dumps(value,ensure_ascii=False)+'\n\n').encode('utf-8')


def test_stream_uses_complete_provider_object_and_never_partial_text():
    complete,_=response();complete['output'][0]['content'][0]['text']='B'
    source=io.BytesIO(b': heartbeat\n\n'+event({'type':'response.output_text.delta','delta':'A'})+
                      event({'type':'response.completed','response':complete}))
    assert _read_response_events(source)==complete


@pytest.mark.parametrize('ending',[b'',b'data: [DONE]\n\n'])
def test_partial_stream_without_terminal_response_is_not_a_judgment(ending):
    with pytest.raises(URLError,match='terminal'):
        _read_response_events(io.BytesIO(event({'type':'response.output_text.delta','delta':'A'})+ending))


def test_stream_truncation_still_returns_usage_but_not_completed_status():
    value,_=response();value['status']='incomplete'
    assert _read_response_events(io.BytesIO(event({'type':'response.incomplete','response':value})))==value
    value['usage']=None
    with pytest.raises(ValueError,match='billing usage'):
        _read_response_events(io.BytesIO(event({'type':'response.incomplete','response':value})))


def test_stream_keeps_original_request_deadline():
    with pytest.raises(TimeoutError,match='request timeout'):
        _read_response_events(io.BytesIO(b': heartbeat\n\n'),deadline=0)


def test_stream_wire_override_is_bound_and_cached_replay_keeps_original_delivery(tmp_path):
    budget=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'training':1});sent=[]
    payload={'model':'gpt-6.1-sol','max_output_tokens':4096,'service_tier':'flex','input':[]}
    def fake(url,wire,key,timeout):sent.append(wire);return response()
    first=BudgetTransport(budget,'training',fake,stream_responses=True).for_call('packet/0/0/0')
    returned=first('https://api.openai.com/v1/responses',payload,'TEST_KEY',900)
    assert sent==[{**payload,'stream':True}] and 'stream' not in payload
    assert first.receipt['request_transport']['protocol']=='responses_sse_v1'
    replay=BudgetTransport(budget,'training',lambda *args:pytest.fail('No duplicate HTTP')).for_call('packet/0/0/0')
    assert replay('https://api.openai.com/v1/responses',payload,'TEST_KEY',900)==returned
    assert replay.receipt['request_transport']==first.receipt['request_transport']
    old=BudgetTransport(budget,'training',lambda *args:response()).for_call('old/0/0/0')
    old('https://api.openai.com/v1/responses',payload,'TEST_KEY',900)
    old_replay=BudgetTransport(budget,'training',lambda *args:pytest.fail('No new mode for old call'),stream_responses=True).for_call('old/0/0/0')
    old_replay('https://api.openai.com/v1/responses',payload,'TEST_KEY',900)
    assert 'request_transport' not in old_replay.receipt


def test_interrupted_stream_retains_unknown_reservation_and_never_replays_http(tmp_path):
    budget=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'training':1})
    payload={'model':'gpt-6.1-sol','max_output_tokens':4096,'service_tier':'flex','input':[]}
    def truncated(*args):return _read_response_events(io.BytesIO(event({'type':'response.output_text.delta','delta':'A'})))
    first=BudgetTransport(budget,'training',truncated,stream_responses=True).for_call('packet/0/0/0')
    with pytest.raises(URLError):first('https://api.openai.com/v1/responses',payload,'TEST_KEY',900)
    before=budget.snapshot()['training'];assert before['unresolved_reservations_usd']>0
    replay=BudgetTransport(budget,'training',lambda *args:pytest.fail('Must not redispatch')).for_call('packet/0/0/0')
    with pytest.raises(URLError):replay('https://api.openai.com/v1/responses',payload,'TEST_KEY',900)
    assert budget.snapshot()['training']==before


def test_real_http_decoder_stream_and_budgeted_packet_binding(tmp_path,monkeypatch):
    from prism import judges
    manifest,budget,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY');sent=[]
    class Reply(io.BytesIO):
        headers={'x-request-id':'streamed-request'}
    class Opener:
        def open(self,request,timeout):
            payload=json.loads(request.data);assert payload['stream'] is True and payload['service_tier']=='flex' and timeout==900
            sent.append(payload)
            value,_=response()
            return Reply(event({'type':'response.output_text.delta','delta':'ignored'})+event({'type':'response.completed','response':value}))
    monkeypatch.setattr(judges,'build_opener',lambda *args:Opener())
    result=run_manifest(manifest,execute=True,stream_responses=True,poll_seconds=.01)
    assert result['state']=='completed' and result['responses_transport']=='sse_v1' and len(sent)==8
    events=rows(tmp_path/'train.results.jsonl');packet=_load_packet(path)
    _,_,accounting=_records(packet,tmp_path/'train.results.jsonl',with_accounting=True)
    assert accounting['actual_judge_calls']==8 and budget.snapshot()['training']['dispatches']==8
    budget.control(True,'Offline inspection after completion')
    assert recover_manifest(manifest)['recovered_records']==0
    events[0]['result']['budget_transport']['request_transport']['effective_payload_sha256']='0'*64
    bad=tmp_path/'bad.jsonl';bad.write_text(''.join(json.dumps(r)+'\n' for r in events),encoding='utf-8')
    with pytest.raises(ValueError,match='streaming request payload'):_records(packet,bad)


def test_cached_streamed_response_missing_jsonl_recovers_with_wire_binding(tmp_path,monkeypatch):
    from prism.judges import APIJudge
    manifest,budget,path=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    packet=_load_packet(path);task=packet['tasks'][0];call=task['calls'][0]
    bound=BudgetTransport(budget,'training',lambda *args:response(),stream_responses=True).for_call(f"{packet['packet_id']}/0/0/0")
    judge=APIJudge({'execution':{'allow_paid_api_calls':True},'judges':{'training':packet['settings']}},'training',transport=bound)
    assert judge.call(task['messages'],task['left'],task['right'],call['seed'],call['flip'])['label'] is not None
    budget.control(True,'Cached streamed response before append')
    assert recover_manifest(manifest)['recovered_records']==1
    restored=rows(tmp_path/'train.results.jsonl')[0]['result']
    assert restored['budget_transport']['request_transport']==bound.receipt['request_transport']
    assert budget.snapshot()['training']['dispatches']==1


def test_streamed_observation_import_and_independent_reward_audit(tmp_path,monkeypatch):
    from prism import pipeline
    from prism.io import root
    from prism.local_judge import run_packet
    from prism.v2 import materialize,audit_reward
    from test_v2 import prepared as study_prepared,fake
    c,profile,data=study_prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    budget=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'training':1})
    for spec in data['packets']:
        p=Path(spec['packet']);target=p.with_name(p.name.replace('.packet.json','.results.jsonl'))
        run_packet(p,target,execute=True,transport=BudgetTransport(budget,'training',fake,stream_responses=True))
    assert materialize(c,profile,root(c)/'local_judge_packets')['passed']
    pipeline.reward(c,'prism',4,11)
    assert audit_reward(c,'prism',4,11)['passed']
