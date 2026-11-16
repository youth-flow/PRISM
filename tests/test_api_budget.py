from concurrent.futures import ThreadPoolExecutor
from threading import Event
from urllib.error import URLError

import pytest

from prism.api_budget import BudgetExceeded, BudgetTransport, USDBudget, usage_micro_usd


def test_usage_includes_cache_writes_and_reasoning_output():
    assert usage_micro_usd('gpt-6.1-sol',{'input_tokens':1000,'output_tokens':200,
        'input_tokens_details':{'cached_tokens':100,'cache_write_tokens':300,
         'cache_write_tokens_details':{'ttl_30m_tokens':200,'ttl_12h_tokens':100}}}) == 4110
    with pytest.raises(ValueError):
        usage_micro_usd('gpt-6.1-sol',{'input_tokens':1,'output_tokens':2,'input_tokens_details':{'cached_tokens':2}})
    assert usage_micro_usd('gpt-6.1-sol',{'input_tokens':10,'output_tokens':2},'flex')==20
    with pytest.raises(ValueError,match='billing tier'):
        usage_micro_usd('gpt-6.1-sol',{'input_tokens':10,'output_tokens':2},'priority')


def test_completed_response_replays_after_missing_jsonl_without_new_http(tmp_path):
    b=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'training':1})
    sent=[]
    def fake(*args):
        sent.append(True)
        return {'usage':{'input_tokens':10,'output_tokens':2},'status':'completed'},'request-1'
    t=BudgetTransport(b,'training',fake)
    payload={'model':'gpt-6.1-sol','max_output_tokens':64,'input':[]}
    first=t.for_call('packet/task/0');second=t.for_call('packet/task/0')
    assert first('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)==second('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)
    assert len(sent)==1 and second.receipt['replayed_response']
    assert b.snapshot()['training']['dispatches']==1
    with pytest.raises(ValueError,match='different request'):
        t.for_call('packet/task/0')('https://api.openai.com/v1/responses',{**payload,'max_output_tokens':128},'TEST_KEY',1)


def test_unknown_dispatch_stays_reserved_and_concurrent_ceiling_is_atomic(tmp_path):
    b=USDBudget(tmp_path/'budget.sqlite',total_usd=.1,purposes={'training':.1})
    payload={'model':'gpt-6.1-sol','max_output_tokens':1000,'input':[]}
    def reserve(i):
        try:b.reserve(str(i),'training','gpt-6.1-sol',payload);return True
        except BudgetExceeded:return False
    with ThreadPoolExecutor(max_workers=16) as pool: outcomes=list(pool.map(reserve,range(30)))
    held=b.snapshot()['training']['unresolved_reservations_usd']
    assert 0<sum(outcomes)<30 and 0<held<=.1
    def fail(*args):raise URLError('network')
    t=BudgetTransport(b,'training',fail)
    # The reserved identity cannot silently dispatch again after a crash.
    reserved_id=str(next(i for i,accepted in enumerate(outcomes) if accepted))
    with pytest.raises(URLError,match='reservation retained'):
        t.for_call(reserved_id)('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)
    with pytest.raises(ValueError,match='contract changed'):
        USDBudget(tmp_path/'budget.sqlite',total_usd=2,purposes={'training':2})


def test_flex_actual_tier_controls_cost_and_inflight_reservations_can_settle(tmp_path):
    b=USDBudget(tmp_path/'budget.sqlite',total_usd=.04,purposes={'training':.04})
    entered,release=Event(),Event()
    calls=[]
    def fake(url,payload,key,timeout):
        calls.append(payload['service_tier'])
        if len(calls)==1:
            entered.set();assert release.wait(5)
        return {'usage':{'input_tokens':10,'output_tokens':2},'service_tier':'default'},'test'
    payload={'model':'gpt-6.1-sol','service_tier':'flex','max_output_tokens':1000,'input':[]}
    t=BudgetTransport(b,'training',fake)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(t.for_call('first'),'https://api.openai.com/v1/responses',payload,'TEST_KEY',5)
        assert entered.wait(5)
        second=pool.submit(t.for_call('second'),'https://api.openai.com/v1/responses',payload,'TEST_KEY',5)
        release.set();first.result();second.result()
    assert calls==['flex','flex']
    # Requesting Flex must not undercount an actual standard-tier response.
    assert b.snapshot()['training']['charged_usd_estimate']==.00008
