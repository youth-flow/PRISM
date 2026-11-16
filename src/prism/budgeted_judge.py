"""Local-only OpenAI packets with one recoverable ledger and explicit controls.

Planning, inspection and cached-response recovery make no HTTP requests. Paid
execution requires both an unpaused ledger and --execute. No key enters a file.
"""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from threading import Event,Lock
import time
from urllib.error import HTTPError,URLError
from urllib.request import Request,build_opener

import yaml

from .api_budget import BudgetExceeded, BudgetTransport, DispatchPaused, USDBudget
from .io import digest, dump, rows
from .judges import APIJudge,_NoRedirect
from .local_judge import _load_packet, _records, _transport_retry_protocol, NETWORK_RETRY_LIMIT, run_packet


def _transport_error_metadata(error):
    """Diagnostic types/codes only: exception strings may contain credentials."""
    value={'exception_type':type(error).__name__}
    reason=getattr(error,'reason',None)
    if reason is not None:value['reason_type']=type(reason).__name__
    cause=reason if isinstance(reason,BaseException) else error
    for name in ('errno','winerror','verify_code'):
        number=getattr(cause,name,None)
        if type(number) is int:value[name]=number
    # ssl exposes stable symbolic codes separately from sensitive error text.
    for name in ('library','reason'):
        code=getattr(cause,name,None)
        if isinstance(code,str) and code and len(code)<=80 and all(ch.isupper() or ch.isdigit() or ch=='_' for ch in code):value['ssl_'+name]=code
    if isinstance(error,HTTPError):value['http_status']=error.code
    return value


@contextmanager
def owner_lock(path):
    """One execution/recovery process per ledger, including Windows."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    if path.stat().st_size==0:stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Another Judge worker owns this ledger; no request was sent') from None
    try:
        yield
    finally:
        stream.seek(0)
        if os.name=='nt':
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def prepare_manifest(packets,profile,ledger,destination,workers=8):
    """Bind existing packets; never create or reset a spending ledger."""
    if type(workers) is not int or not 1<=workers<=16:
        raise ValueError('Workers per packet must be in 1..16')
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Preserve the existing run manifest')
    specs=[]
    for path in map(lambda p:Path(p).resolve(),packets):
        packet=_load_packet(path)
        result=path.with_name(path.name.removesuffix('.packet.json')+'.results.jsonl')
        if result==path:raise ValueError('Use *.packet.json filenames')
        specs.append({'packet':str(path),'results':str(result),'packet_sha256':digest(path)})
    manifest={'kind':'prism_budgeted_judge','version':1,'profile':str(Path(profile).resolve()),
              'profile_sha256':digest(profile),'ledger':str(Path(ledger).resolve()),'packets':specs,
              'workers_per_packet':workers,'max_total_workers':32,
              'status':str(destination.with_suffix('.status.json'))}
    # Check all inputs without touching the ledger or credentials.
    _validate_manifest(manifest)
    dump(destination,manifest)
    return manifest


def _validate_manifest(m):
    if m.get('kind')!='prism_budgeted_judge' or m.get('version')!=1:
        raise ValueError('Unknown budgeted Judge manifest')
    if digest(m['profile'])!=m['profile_sha256']:raise ValueError('Judge profile changed')
    profile=yaml.safe_load(Path(m['profile']).read_text(encoding='utf-8'))
    money=profile['spending_usd']
    if money['user_total']!=40 or money['operational_total']>39:
        raise ValueError('V2 requires the frozen USD40 user / USD39 operational ceiling')
    workers=m['workers_per_packet']
    if type(workers) is not int or not 1<=workers<=16 or m['max_total_workers']!=32:
        raise ValueError('Invalid frozen concurrency limits')
    loaded=[];identities=set();paths=set();roles=set()
    for spec in m['packets']:
        path=Path(spec['packet']);result=Path(spec['results'])
        if digest(path)!=spec['packet_sha256']:raise ValueError('Judge packet bytes changed')
        p=_load_packet(path)
        if p['packet_id'] in identities or result.resolve() in paths:
            raise ValueError('Duplicate packet or output in one manifest')
        identities.add(p['packet_id']);paths.add(result.resolve());roles.add(p['role'])
        if p['settings']!=profile['judges'][p['role']] or p['settings']['backend']!='openai':
            raise ValueError('Packet does not match the frozen OpenAI Judge profile')
        if result.exists():_records(p,result)
        loaded.append((spec,p))
    if not loaded or len(roles)!=1:raise ValueError('Use one nonempty training or evaluation batch')
    return profile,loaded


def load_manifest(path):
    m=json.loads(Path(path).read_text(encoding='utf-8'))
    profile,loaded=_validate_manifest(m)
    return m,profile,loaded


def _ledger(m,profile):
    b=USDBudget.open_existing(m['ledger']);state=b.state();money=profile['spending_usd']
    expected={'total':money['operational_total'],**{k:money[k] for k in ('calibration','training','evaluation')}}
    if any(state['limits_usd'].get(k,float('inf'))>v for k,v in expected.items()):
        raise ValueError('Existing live USD ceilings exceed the final profile; explicitly lower/reallocate first')
    return b


def _progress(loaded):
    result={}
    for spec,p in loaded:
        data=[];path=Path(spec['results'])
        if path.exists():
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if line.endswith('\n'):data.append(json.loads(line))
        result[p['packet_id']]={'completed':len({r['task_id'] for r in data if r['result']['label'] is not None}),
                                'planned':len(p['tasks']),'recorded_calls':len(data),
                                'invalid_calls':sum(r['result']['label'] is None for r in data)}
    return result


def inspect_manifest(path):
    m,profile,loaded=load_manifest(path);b=_ledger(m,profile)
    reports=[]
    with b.transaction() as db:
        for spec,p in loaded:
            recorded={r['result'].get('budget_transport',{}).get('dispatch_id') for r in (rows(spec['results']) if Path(spec['results']).exists() else [])}
            known=db.execute('SELECT id,status FROM dispatch WHERE id LIKE ?',(p['packet_id']+'/%',)).fetchall()
            cached_states=('completed','failed_no_charge')
            reports.append({'packet_id':p['packet_id'],'cached_missing_jsonl':sum(i not in recorded and s in cached_states for i,s in known),
                            'unresolved_missing_jsonl':sum(i not in recorded and s not in cached_states for i,s in known)})
    return {'packets':_progress(loaded),'ledger':b.state(),'recovery':reports,'http_requests_sent':0}


def _recover(m,loaded,b):
    """Append only already-dispatched attempts; unknown responses stay failures."""
    recovered=0
    def forbidden(*args):raise AssertionError('Offline recovery must never dispatch HTTP')
    for spec,p in loaded:
        _,indexed=_records(p,spec['results']);transport=BudgetTransport(b,p['role'],forbidden)
        with b.transaction() as db:
            known_dispatches=dict(db.execute('SELECT id,status FROM dispatch WHERE id LIKE ?',(p['packet_id']+'/%',)).fetchall())
        def recover_call(task,attempt,retry):
            nonlocal recovered
            call=task['calls'][attempt];identity=f"{p['packet_id']}/{task['id']}/{attempt}/{retry}"
            # No HTTP: only identities already present in the original ledger.
            env=p['settings']['api_key_env'];before=os.environ.get(env)
            os.environ[env]='OFFLINE_CACHED_RESPONSE_REPLAY'
            try:
                bound=transport.for_call(identity)
                judge=APIJudge({'execution':{'allow_paid_api_calls':True},'judges':{p['role']:p['settings']}},p['role'],transport=bound)
                value=judge.call(task['messages'],task['left'],task['right'],call['seed'],call['flip'])
            finally:
                if before is None:os.environ.pop(env,None)
                else:os.environ[env]=before
            value['budget_transport']=bound.receipt
            value['recovery']={'offline':True,'reason':'received_response_or_unknown_dispatch_recovered_from_ledger'}
            if value['label'] is None and (value.get('error') or {}).get('type')=='network_error':
                value['error']={'type':'response_unavailable','retryable':True};b.uncertain(identity)
            record={'packet_id':p['packet_id'],'task_id':task['id'],'attempt':attempt,'result':value}
            if retry:record['transport_retry']=retry
            target=Path(spec['results']);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            selected={**value,'transport_retry':retry} if retry else value
            indexed[(task['id'],attempt)]=selected;recovered+=1
            return selected
        for task in p['tasks']:
            for attempt in range(len(task['calls'])):
                previous=indexed.get((task['id'],attempt))
                if previous is None:
                    if f"{p['packet_id']}/{task['id']}/{attempt}/0" not in known_dispatches:break
                    previous=recover_call(task,attempt,0)
                if previous['label'] is not None:break
                if attempt==len(task['calls'])-1 and _transport_retry_protocol(p):
                    for retry in range(previous.get('transport_retry',0)+1,NETWORK_RETRY_LIMIT+1):
                        if f"{p['packet_id']}/{task['id']}/{attempt}/{retry}" not in known_dispatches:break
                        previous=recover_call(task,attempt,retry)
                        if previous['label'] is not None:break
        _records(p,spec['results'])
    return recovered


def recover_manifest(path):
    m,profile,loaded=load_manifest(path);b=_ledger(m,profile)
    with owner_lock(str(b.path)+'.worker.lock'):
        if not b.state()['paused']:raise RuntimeError('Pause/drain execution before offline recovery')
        count=_recover(m,loaded,b)
    return {'recovered_records':count,'http_requests_sent':0,'inspection':inspect_manifest(path)}


def run_manifest(path,*,execute=False,transport=None,poll_seconds=20,network_retries=0,stream_responses=False):
    if not execute:raise RuntimeError('Paid Judge execution requires --execute')
    m,profile,loaded=load_manifest(path);b=_ledger(m,profile)
    if type(network_retries) is not int or not 0<=network_retries<=NETWORK_RETRY_LIMIT:
        raise ValueError('Network retransmissions must stay within 0..2')
    if network_retries and not all(_transport_retry_protocol(p) for _,p in loaded):
        raise ValueError('Network retransmissions are restricted to checked training packets')
    if type(stream_responses) is not bool or (stream_responses and not all(_transport_retry_protocol(p) and p['settings']['backend']=='openai' for _,p in loaded)):
        raise ValueError('Streaming is restricted to checked OpenAI training packets')
    with owner_lock(str(b.path)+'.worker.lock'):
        if b.state()['paused']:raise DispatchPaused('Ledger is paused; explicitly resume after approval')
        _recover(m,loaded,b)
        abort=Event();started=time.monotonic();network_failures=deque();network_lock=Lock();failures=[]
        class ManagedTransport(BudgetTransport):
            def for_call(self,identity):
                inner=super().for_call(identity)
                class Bound:
                    @property
                    def receipt(self):return inner.receipt
                    def remember(self,error):
                        diagnostic=_transport_error_metadata(error)
                        if inner.receipt is not None:inner.receipt['transport_failure']=diagnostic
                        with network_lock:failures.append({'dispatch_id':identity,'timestamp_utc':datetime.now(timezone.utc).isoformat(),**diagnostic})
                    def __call__(self,url,payload,key,timeout):
                        if abort.is_set():raise DispatchPaused('Another packet stopped; no new dispatch')
                        attempt=int(identity.split('/')[-2])
                        if attempt:time.sleep(min(20,2**attempt))
                        if abort.is_set():raise DispatchPaused('Another packet stopped; no new dispatch')
                        try:return inner(url,payload,key,timeout)
                        except HTTPError as error:
                            self.remember(error)
                            if error.code!=429 and error.code<500:abort.set()
                            raise
                        except (URLError,TimeoutError,ConnectionError) as error:
                            self.remember(error)
                            # Isolated packet loss uses only the remaining
                            # original slots. Two losses in 60 seconds trip the
                            # shared breaker before a burst consumes all slots.
                            # Already admitted HTTP requests still drain/log.
                            with network_lock:
                                now=time.monotonic();network_failures.append(now)
                                while network_failures[0]<now-60:network_failures.popleft()
                                if len(network_failures)>=2:abort.set()
                            raise
                        except BaseException as error:
                            self.remember(error)
                            abort.set();raise
                return Bound()
        role=loaded[0][1]['role'];managed=ManagedTransport(b,role,transport,stream_responses=stream_responses)
        def status(state,error=None):
            with network_lock:observed=list(failures)
            value={'state':state,'pid':os.getpid(),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                   'elapsed_seconds':time.monotonic()-started,'packets':_progress(loaded),'ledger':b.state(),
                   'network_retries_limit':network_retries,
                   'network_breaker':{'errors':2,'window_seconds':60},
                   'responses_transport':'sse_v1' if stream_responses else 'json',
                   'transport_failures':observed,
                   'code_sha256':{f:digest(Path(__file__).with_name(f)) for f in ('budgeted_judge.py','api_budget.py','local_judge.py')},
                   'error_type':type(error).__name__ if error else None}
            dump(m['status'],value)
            return value
        status('running')
        try:
            groups=min(len(loaded),m['max_total_workers']//m['workers_per_packet'])
            with ThreadPoolExecutor(max_workers=groups) as executor:
                futures={executor.submit(run_packet,s['packet'],s['results'],execute=True,transport=managed,workers=m['workers_per_packet'],network_retries=network_retries) for s,p in loaded}
                try:
                    while futures:
                        done,futures=wait(futures,timeout=poll_seconds,return_when=FIRST_COMPLETED)
                        for item in done:item.result()
                        status('running')
                except BaseException:
                    abort.set()
                    if not b.state()['paused']:b.control(True,'Worker stopped; inspect saved results before resuming')
                    raise
            final=status('completed')
            for spec,p in loaded:
                _,_,accounting=_records(p,spec['results'],with_accounting=True)
                dump(Path(spec['results']).with_suffix('.receipt.json'),
                     {'packet_id':p['packet_id'],'input_sha256':{spec['packet']:digest(spec['packet']),spec['results']:digest(spec['results'])},'accounting':accounting,
                      'network_retries_limit':network_retries,'code_sha256':final['code_sha256']})
            return final
        except BaseException as error:
            status('paused' if isinstance(error,DispatchPaused) else 'budget_limit' if isinstance(error,BudgetExceeded) else 'failed',error)
            raise


_WORKER_PAUSE_REASON='Worker stopped; inspect saved results before resuming'


def _model_healthcheck(settings):
    key=os.environ.get(settings['api_key_env'])
    if not key:return False
    try:
        with build_opener(_NoRedirect()).open(Request('https://api.openai.com/v1/models',headers={'Authorization':'Bearer '+key}),timeout=15) as response:
            return response.status==200
    except (OSError,HTTPError,URLError):return False


def _only_network_pause(status,budget_state):
    failures=status.get('transport_failures') or []
    network={'URLError','TimeoutError','ConnectionError','ConnectionResetError','RemoteDisconnected','SSLError','SSLEOFError'}
    recoverable=lambda value: (value.get('exception_type') in network or
        (value.get('exception_type')=='HTTPError' and (value.get('http_status')==429 or value.get('http_status',0)>=500)))
    return (status.get('state')=='paused' and status.get('error_type')=='DispatchPaused'
            and status.get('pid')==os.getpid() and budget_state['paused']
            and budget_state['pause_reason']==_WORKER_PAUSE_REASON
            and any(f.get('exception_type') in network for f in failures)
            and all(recoverable(f) for f in failures))


def _resume_own_network_pause(budget):
    # Atomic compare-and-set: never overwrite a user/operator pause racing with
    # the health check. Existing budget triggers and all costs stay unchanged.
    with budget.transaction() as db:
        cursor=db.execute('UPDATE dispatch_control SET paused=0,reason=? WHERE id=1 AND paused=1 AND reason=?',
                          ('Checked bounded network recovery; original attempts and caps retained',_WORKER_PAUSE_REASON))
        if cursor.rowcount!=1:raise DispatchPaused('Explicit pause changed; no automatic resume')
        db.execute('INSERT INTO admission_events(action,reason) VALUES (?,?)',
                   ('resume','Checked bounded network recovery; original attempts and caps retained'))


def supervise_manifest(path,*,execute=False,transport=None,poll_seconds=20,network_retries=0,max_network_resumes=0,stream_responses=False,healthcheck=None,sleeper=time.sleep):
    """Bounded orchestration only; run_packet still enforces every original slot.

    No restart on budget stops, authentication, exhausted semantic/transport
    slots, internal errors or explicit pauses. Every stopped status is archived.
    """
    if not execute:raise RuntimeError('Paid Judge execution requires --execute')
    if type(max_network_resumes) is not int or not 0<=max_network_resumes<=8:
        raise ValueError('Network resumes must stay within 0..8')
    if max_network_resumes==0:
        return run_manifest(path,execute=True,transport=transport,poll_seconds=poll_seconds,network_retries=network_retries,stream_responses=stream_responses)
    m,profile,loaded=load_manifest(path);b=_ledger(m,profile)
    if max_network_resumes and any(p['role']!='training' for _,p in loaded):
        raise ValueError('Automatic network recovery is restricted to checked training')
    def available():
        return _model_healthcheck(loaded[0][1]['settings'])
    healthcheck=healthcheck or available
    for resumed in range(max_network_resumes+1):
        try:
            return run_manifest(path,execute=True,transport=transport,poll_seconds=poll_seconds,network_retries=network_retries,stream_responses=stream_responses)
        except DispatchPaused:
            if resumed==max_network_resumes:raise
            if b.state()['pause_reason']!=_WORKER_PAUSE_REASON or not Path(m['status']).exists():raise
            status=json.loads(Path(m['status']).read_text(encoding='utf-8'))
            if not _only_network_pause(status,b.state()):raise
            archive=Path(m['status']).parent/'network_pause_history'
            archive.mkdir(exist_ok=True)
            target=archive/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
            dump(target,{'stopped_status':status,'manifest_sha256':digest(path),'next_network_resume':resumed+1,
                         'max_network_resumes':max_network_resumes,'new_semantic_slots':0,'http_requests_sent_by_recovery':0})
            repair=recover_manifest(path)
            if any(r['cached_missing_jsonl'] or r['unresolved_missing_jsonl'] for r in repair['inspection']['recovery']):raise
            sleeper(30)
            if not healthcheck():raise DispatchPaused('Network health check failed; ledger stays paused') from None
            _resume_own_network_pause(b)
            print(json.dumps({'network_resume':resumed+1,'original_request_limits_unchanged':True,
                              'archived_status':str(target),'offline_recovery_records':repair['recovered_records']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--packet',action='append',required=True)
    prepare.add_argument('--profile',required=True);prepare.add_argument('--ledger',required=True);prepare.add_argument('--output',required=True)
    prepare.add_argument('--workers',type=int,default=8)
    init=sub.add_parser('init-ledger');init.add_argument('--ledger',required=True);init.add_argument('--profile',required=True)
    state=sub.add_parser('state');state.add_argument('--ledger',required=True)
    for name in ('inspect','recover','run'):
        p=sub.add_parser(name);p.add_argument('--manifest',required=True)
        if name=='run':
            p.add_argument('--execute',action='store_true');p.add_argument('--network-retries',type=int,default=0)
            p.add_argument('--max-network-resumes',type=int,default=0)
            p.add_argument('--responses-stream',action='store_true')
    for name in ('pause','resume'):
        p=sub.add_parser(name);p.add_argument('--ledger',required=True);p.add_argument('--reason',required=True)
    a=parser.parse_args()
    if a.action=='prepare':result=prepare_manifest(a.packet,a.profile,a.ledger,a.output,a.workers)
    elif a.action=='init-ledger':
        if Path(a.ledger).exists():raise FileExistsError('Never reset an existing spending ledger')
        money=yaml.safe_load(Path(a.profile).read_text(encoding='utf-8'))['spending_usd']
        if money['user_total']!=40 or money['operational_total']>39:raise ValueError('V2 requires the USD40/39 contract')
        b=USDBudget(a.ledger,total_usd=money['operational_total'],purposes={k:money[k] for k in ('calibration','training','evaluation')})
        b.control(True,'New ledger starts paused until explicit authorization');result=b.state()
    elif a.action=='state':result=USDBudget.open_existing(a.ledger).state()
    elif a.action in ('pause','resume'):
        b=USDBudget.open_existing(a.ledger);b.control(a.action=='pause',a.reason);result=b.state()
    elif a.action=='inspect':result=inspect_manifest(a.manifest)
    elif a.action=='recover':result=recover_manifest(a.manifest)
    else:result=supervise_manifest(a.manifest,execute=a.execute,network_retries=a.network_retries,max_network_resumes=a.max_network_resumes,stream_responses=a.responses_stream)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
