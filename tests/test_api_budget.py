from concurrent.futures import ThreadPoolExecutor
from threading import Event
import sqlite3
from urllib.error import URLError

import pytest

from prism.api_budget import BudgetExceeded, BudgetTransport, DispatchPaused, USDBudget, journal_mode, usage_micro_usd


@pytest.mark.parametrize('version,expected',[((3,50,0),'DELETE'),((3,50,7),'WAL'),((3,44,6),'WAL'),((3,45,0),'DELETE'),((3,51,2),'DELETE'),((3,51,3),'WAL')])
def test_unpatched_sqlite_does_not_enable_wal(version,expected):
    assert journal_mode(version)==expected


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


def test_lowered_live_cap_applies_to_old_workers_and_preserves_uncertain_spend(tmp_path):
    path=tmp_path/'budget.sqlite'
    old=USDBudget(path,total_usd=1,purposes={'training':1})
    fresh=USDBudget(path,total_usd=1,purposes={'training':1})
    payload={'model':'gpt-6.1-sol','max_output_tokens':1000,'input':[]}
    old.reserve('unknown','training','gpt-6.1-sol',payload)
    old.uncertain('unknown')
    before=old.snapshot()
    fresh.tighten_limits(total_usd=.06,purposes={'training':.06},reason='User lowers budget')
    # The old object still holds its original $1 contract. Database enforcement
    # prevents a process from continuing at that stale limit before any HTTP.
    old.reserve('allowed','training','gpt-6.1-sol',payload)
    with pytest.raises(BudgetExceeded,match='ceiling'):
        old.reserve('blocked','training','gpt-6.1-sol',payload)
    # A pre-upgrade process doing a direct INSERT is independently constrained.
    with old.transaction() as db:
        with pytest.raises(sqlite3.IntegrityError,match='Live USD ceiling'):
            db.execute("INSERT INTO dispatch VALUES ('old-code','training','gpt-6.1-sol','test',40000,NULL,'reserved',NULL,NULL)")
    assert old.snapshot()['training']['dispatches']==2
    assert old.snapshot()['training']['unresolved_reservations_usd']>before['training']['unresolved_reservations_usd']
    with pytest.raises(ValueError,match='only tighten'):
        fresh.tighten_limits(total_usd=.07,purposes={'training':.06},reason='No silent increases')
    with pytest.raises(BudgetExceeded,match='already charged/reserved'):
        fresh.tighten_limits(total_usd=.01,purposes={'training':.01},reason='Cannot erase spending')


def test_purpose_reallocation_keeps_total_cap_and_pending_wait_semantics(tmp_path):
    b=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'training':.8,'evaluation':.2})
    b.tighten_limits(total_usd=.1,purposes={'training':.05,'evaluation':.05},reason='Lower total')
    revised=b.reallocate_limits(purposes={'training':.08,'evaluation':.02},reason='User fixes counts; discounted evaluation')
    assert revised=={'total':.1,'training':.08,'evaluation':.02}
    payload={'model':'gpt-6.1-sol','max_output_tokens':1000,'input':[]}
    for i in range(2):b.reserve(str(i),'training','gpt-6.1-sol',payload)
    with pytest.raises(BudgetExceeded) as exc:b.reserve('waiting','training','gpt-6.1-sol',payload)
    assert exc.value.waitable
    for i in range(2):b.complete(str(i),'gpt-6.1-sol',{'usage':{'input_tokens':10,'output_tokens':2}},'test')
    b.reserve('waiting','training','gpt-6.1-sol',payload)
    with pytest.raises(ValueError,match='only tighten'):
        b.tighten_limits(total_usd=.11,purposes={'training':.08,'evaluation':.02},reason='Must not raise total')


def test_received_http_survives_settlement_failure_without_another_request(tmp_path,monkeypatch):
    b=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'training':1})
    payload={'model':'gpt-6.1-sol','max_output_tokens':64,'input':[]};sent=[]
    def fake(*args):
        sent.append(True)
        return {'usage':{'input_tokens':10,'output_tokens':2},'service_tier':'flex'},'received'
    transport=BudgetTransport(b,'training',fake);complete=b.complete
    def blocked(*args):raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(b,'complete',blocked)
    with pytest.raises(sqlite3.OperationalError,match='database is locked'):
        transport.for_call('same')('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)
    assert b.cached_response('same','gpt-6.1-sol',payload) is not None
    monkeypatch.setattr(b,'complete',complete)
    bound=transport.for_call('same')
    assert bound('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)[1]=='received'
    assert len(sent)==1 and bound.receipt['replayed_response']
    assert b.snapshot()['training']['charged_usd_estimate']==.00002


def test_32_workers_on_existing_large_ledger_settle_without_lost_responses(tmp_path):
    b=USDBudget(tmp_path/'budget.sqlite',total_usd=20,purposes={'training':20})
    # The real incident occurred after thousands of rows, not on an empty DB.
    with b.transaction() as db:
        db.executemany("INSERT INTO dispatch VALUES (?,'training','gpt-6.1-sol','old',1,0,'completed','{}',NULL)",
                       [(f'old-{i}',) for i in range(12000)])
    reopened=USDBudget.open_existing(b.path)
    payload={'model':'gpt-6.1-sol','max_output_tokens':64,'input':[]}
    def fake(*args):return {'usage':{'input_tokens':10,'output_tokens':2},'service_tier':'flex'},'test'
    transports=[BudgetTransport(b,'training',fake),BudgetTransport(reopened,'training',fake)]
    def call(i):return transports[i%2].for_call(str(i))('https://api.openai.com/v1/responses',payload,'TEST_KEY',5)
    with ThreadPoolExecutor(max_workers=32) as pool:completed=list(pool.map(call,range(128)))
    assert len(completed)==128
    snapshot=b.snapshot()['training']
    assert snapshot['dispatches']==12128 and snapshot['unresolved_reservations_usd']==0
    assert snapshot['charged_usd_estimate']==.00256


def test_pause_blocks_new_spending_but_preserves_cached_replay(tmp_path):
    b=USDBudget(tmp_path/'budget.sqlite',total_usd=1,purposes={'training':1})
    payload={'model':'gpt-6.1-sol','max_output_tokens':64,'input':[]}
    t=BudgetTransport(b,'training',lambda *args:({'usage':{'input_tokens':10,'output_tokens':2}},'test'))
    t.for_call('done')('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)
    b.control(True,'User wants code/configuration review first')
    assert b.state()['paused']
    assert t.for_call('done')('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)[1]=='test'
    with pytest.raises(DispatchPaused,match='paused'):
        t.for_call('new')('https://api.openai.com/v1/responses',payload,'TEST_KEY',1)
    assert b.snapshot()['training']['dispatches']==1
    b.control(False,'Explicit subsequent start')
    assert not b.state()['paused']
