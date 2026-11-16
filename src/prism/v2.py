"""Reviewable V2 plans and offline artifact transitions; never calls an API.

Heavy transitions run through the Yancheng CPU wrapper. The original experiment
contract stays immutable; the final execution profile only changes evaluation
transport. Paid work is exclusively prism.budgeted_judge on the local host.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .config import plan_summary
from .io import cell, digest, dump, load_config, receipt, root, rows, verify_receipt
from .local_judge import _execution_settings, _records, export_packet, import_results, validate_packet
from .judges import judge_settings


def execution_profile(c,path):
    profile=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for role in ('training','evaluation'):
        base=judge_settings(c,role);effective=profile['judges'][role]
        overrides={k:v for k,v in effective.items() if base.get(k)!=v}
        if _execution_settings(base,role,overrides)!=effective:
            raise ValueError('Execution profile changed a research Judge parameter')
    money=profile['spending_usd']
    if money['user_total']!=40 or money['operational_total']>39:
        raise ValueError('V2 API budget must stay within USD40, operationally USD39')
    return profile


def freeze_config(template,asset_root,run_root,destination):
    """Resolve verified assets outside an immutable source release; no compute."""
    asset_root=Path(asset_root).resolve();destination=Path(destination).resolve()
    c=yaml.safe_load(Path(template).read_text(encoding='utf-8'))
    for role in ('design','policy','evaluation'):
        c['choices']['prompts'][role]=str(asset_root/'data/small'/f'{role}.jsonl')
    c['choices']['benchmark'].update(path=str(asset_root/'data/small/ifeval.jsonl'),official_source_path=str(asset_root/'assets/ifeval'))
    c['data']['manifest_path']=str(asset_root/'data/small/data.manifest.json')
    c['benchmark']['nltk_data_path']=str(asset_root/'assets/nltk_data')
    c['output']=str(Path(run_root).resolve())
    from .config import load_v2
    validated,missing=load_v2(destination,c,ready=False)
    text=yaml.safe_dump(c,sort_keys=False,allow_unicode=True)
    if destination.exists():
        if destination.read_text(encoding='utf-8')!=text:raise FileExistsError('Frozen config differs; preserve it')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(text,encoding='utf-8',newline='\n')
    return {'config':str(destination),'sha256':digest(destination),'missing_inputs':missing,'api_calls_sent':0,'slurm_jobs_submitted':0}


def describe(c,profile_path):
    profile=execution_profile(c,profile_path)
    counts=plan_summary(c)
    stages=[
        {'phase':1,'work':'Freeze inputs, verified SIF/model; fixed candidates with certified FP32 scoring; fresh FP32 SFT evaluation',
         'resources':'Existing candidates can be reused; SFT generation one GPU, 2 CPU/16G'},
        {'phase':2,'work':'FP32 b/K and original q-space certificate; all query prefixes; four max-budget training packets',
         'resources':'Design/import 16 CPU/24G; sample/export CPU only'},
        {'phase':3,'work':'Local Sol labels, offline cluster import/prefix reuse, independent BT fits and numerical audits',
         'resources':'API at most 32 concurrent locally; Reward 8 CPU/12G per cell; no GPU for API waits'},
        {'phase':4,'work':'Fresh PPO for every method/budget/seed; independent policy audit',
         'resources':'One GPU + 8 CPU/64G per PPO; independent units use additional free GPUs'},
        {'phase':5,'work':'Policy answers and official IFEval; generation receipts and matched sources',
         'resources':'One GPU + 2 CPU/16G per generation unit; CPU artifact audit'},
        {'phase':6,'work':'Local Luna AB/BA/TIE; offline import; KL/IFEval and paired aggregate',
         'resources':'API local without GPU; CPU import/scoring/aggregation'},
    ]
    return {'research_config':c,'effective_judges':profile['judges'],'spending_usd':profile['spending_usd'],
            'counts':counts,'phases':stages,'api_calls_sent':0,'slurm_jobs_submitted':0,
            'ppo_actual_steps_per_cell':counts['ppo_rollout_updates_per_cell']*c['ppo']['num_ppo_epochs']*c['ppo']['num_mini_batches']}


def sample_export(c,profile_path):
    if c['sampling']['budget_coupling']!='nested_prefix':raise ValueError('V2 preparation requires nested prefixes')
    profile=execution_profile(c,profile_path);p=root(c)
    verify_receipt(p/'pool.receipt.json');verify_receipt(p/'design.receipt.json')
    from .pipeline import sample
    packets=p/'local_judge_packets';packets.mkdir(exist_ok=True)
    dependencies=[p/'pool.receipt.json',p/'design.receipt.json',Path(profile_path).resolve()]
    records=[]
    with np.load(p/'design.npz',allow_pickle=False) as z:
        index=z['index'].copy();distributions={m:z[m].copy() for m in c['methods']}
    for method in c['methods']:
        for seed in c['seeds']:
            expected=np.random.default_rng(seed).choice(len(index),size=max(c['budgets']),replace=True,p=distributions[method])
            prefixes={}
            for budget in sorted(c['budgets']):
                out=cell(c,method,budget,seed)
                if (out/'queries.receipt.json').exists():verify_receipt(out/'queries.receipt.json')
                else:sample(c,method,budget,seed)
                q=rows(out/'queries.jsonl');prefixes[budget]=q
                if len(q)!=budget:raise ValueError('Wrong query count')
                for k,row in enumerate(q):
                    edge=int(expected[k]);i,a,b=map(int,index[edge])
                    if row!={'draw_id':k,'edge_id':edge,'prompt_index':i,'a':a,'b':b,'probability':float(distributions[method][edge])}:
                        raise ValueError('Independent draw/order/probability replay failed')
                dependencies.append(out/'queries.receipt.json')
            if any(q!=prefixes[max(prefixes)][:n] for n,q in prefixes.items()):raise ValueError('Query prefixes changed')
            target=packets/f'{method}-n{max(c["budgets"])}-seed{seed}.packet.json'
            if target.exists():
                packet,_=validate_packet(c,target)
                if packet['settings']!=profile['judges']['training']:raise ValueError('Existing training packet has another Judge')
            else:
                export_packet(c,'label',method,max(c['budgets']),seed,target,execution_profile=profile_path)
            dependencies.append(target)
            records.append({'method':method,'seed':seed,'budget':max(c['budgets']),'packet':str(target),'sha256':digest(target)})
    summary={'passed':True,'packets':records,'counts':plan_summary(c),'api_calls_sent':0,'prefixes_verified':True}
    dump(p/'v2.prepared.json',summary)
    receipt(p/'v2.prepared.receipt.json',dependencies+[p/'v2.prepared.json'])
    return summary


def materialize(c,profile_path,results_dir):
    profile=execution_profile(c,profile_path);p=root(c);ready=[]
    # Validate all max streams before publishing any smaller-budget snapshot.
    for method in c['methods']:
        for seed in c['seeds']:
            path=p/'local_judge_packets'/f'{method}-n{max(c["budgets"])}-seed{seed}.packet.json'
            packet,tasks=validate_packet(c,path)
            if packet['settings']!=profile['judges']['training']:raise ValueError('Wrong training Judge')
            result=Path(results_dir)/path.name.replace('.packet.json','.results.jsonl')
            _,indexed=_records(packet,result)
            if not all(any(indexed.get((t['id'],a),{}).get('label') is not None for a in range(len(t['calls']))) for t in tasks):
                raise RuntimeError('Max-budget stream incomplete; no substituted/missing labels allowed')
            ready.append((method,seed,path,result,packet))
    from .pipeline import label
    def no_api(*args):raise AssertionError('Prefix materialization must make no API request')
    dependencies=[Path(profile_path).resolve()]
    for method,seed,path,result,packet in ready:
        import_results(c,path,result)
        maximum=rows(cell(c,method,max(c['budgets']),seed)/'labels.jsonl')
        events=list(packet['prior_calls'])+rows(result)
        for budget in sorted(c['budgets']):
            label(c,method,budget,seed,judge_factory=no_api)
            out=cell(c,method,budget,seed)
            if rows(out/'labels.jsonl')!=maximum[:budget]:raise ValueError('Labels do not match the observation prefix')
            prefix=[event for event in events if int(event['task_id'])<budget]
            logical=len({(event['task_id'],event['attempt']) for event in prefix})
            canonical=len(rows(out/'judge_calls.jsonl'))
            marker=out/'labels.receipt.json';metadata=verify_receipt(marker)
            metadata.update(actual_judge_calls=len(prefix),invalid_calls=sum(event['result']['label'] is None for event in prefix),
                            canonical_judge_calls=canonical,collapsed_transport_calls=len(prefix)-logical,
                            superseded_semantic_calls=logical-canonical,
                            max_transport_retry_used=max((event.get('transport_retry',0) for event in prefix),default=0),
                            raw_api_result_path=str(result.resolve()),source_packet_path=str(path.resolve()))
            metadata['input_sha256'].update({str(result.resolve()):digest(result),str(path.resolve()):digest(path)})
            dump(marker,metadata)
            dependencies.append(out/'labels.receipt.json')
        stream=root(c)/'label_streams'/method/f'seed{seed}'/'judge_calls.jsonl'
        stream_marker=stream.parent/'labels.receipt.json'
        receipt(stream_marker,[stream,path,result],raw_api_result_path=str(result.resolve()),actual_judge_calls=len(events))
        dependencies.append(stream_marker)
        dependencies.extend([path,result])
    summary={'passed':True,'training_valid_observations':plan_summary(c)['training_valid_labels'],'cells':plan_summary(c)['training_cells'],'api_calls_sent':0}
    dump(p/'v2.labels.audit.json',summary)
    receipt(p/'v2.labels.audit.receipt.json',dependencies+[p/'v2.labels.audit.json'])
    return summary


def _audit_openai_observations(c,method,budget,seed,labels,metadata):
    """Independently replay raw binary direction and bounded final-slot retries."""
    raw_path=Path(metadata['raw_api_result_path']);packet_path=Path(metadata['source_packet_path'])
    packet=json.loads(packet_path.read_text(encoding='utf-8'));events=list(packet['prior_calls'])+rows(raw_path)
    if packet['settings']!=judge_settings(c,'training'):raise ValueError('Raw training Judge changed')
    method_index={'empirical':0,'prism':2}[method]
    logical={};counts={};selected={};retry_calls=0;raw_calls=0
    for event in events:
        draw=int(event['task_id'])
        if draw>=budget:continue
        raw_calls+=1;attempt=event['attempt'];retry=event.get('transport_retry',0);value=event['result'];key=(draw,attempt)
        if type(attempt) is not int or not 0<=attempt<3 or type(retry) is not int or not 0<=retry<=2:
            raise ValueError('Raw training attempt/retransmission bound changed')
        if retry:
            previous=logical.get(key);error=(previous or {}).get('error') or {}
            if (attempt!=2 or previous is None or previous['label'] is not None or error.get('retryable') is not True
                    or error.get('type') not in ('network_error','response_unavailable','http_error')
                    or retry!=counts[key]+1):raise ValueError('Raw final-slot retransmission was not a network failure')
            retry_calls+=1
        elif key in logical or attempt!=sum(k[0]==draw for k in logical):
            raise ValueError('Raw semantic attempts are not a contiguous failure prefix')
        parts=[seed,method_index,draw,attempt,991]
        if c['sampling']['budget_coupling']=='independent':parts.insert(1,budget)
        expected=int(np.random.SeedSequence(parts).generate_state(1)[0]);flip=bool(np.random.default_rng(expected).integers(2))
        shown={'A':1,'B':0}.get(value['raw'].strip()) if value.get('response_status')=='completed' else None
        decoded=1-shown if shown is not None and flip else shown
        if (value['seed']!=expected or value['flip']!=flip or value['label']!=decoded
                or value.get('backend')!='openai' or value.get('configured_model')!=packet['settings']['model']
                or value.get('request_endpoint')!='https://api.openai.com/v1/responses'):
            raise ValueError('Independent raw OpenAI direction/seed/identity mismatch')
        dispatch=(value.get('budget_transport') or {}).get('dispatch_id')
        if dispatch is not None and dispatch!=f"{packet['packet_id']}/{draw}/{attempt}/{retry}":
            raise ValueError('Raw budget dispatch identity changed')
        if draw in selected:raise ValueError('An original request followed a completed preference')
        logical[key]=value;counts[key]=retry
        if decoded is not None:selected[draw]=(attempt,retry,value)
    if set(selected)!=set(range(budget)) or metadata['actual_judge_calls']!=raw_calls:
        raise ValueError('Raw training coverage or actual HTTP count changed')
    for draw,label in enumerate(labels):
        attempt,retry,value=selected[draw]
        if label['attempt']!=attempt or label.get('transport_retry',0)!=retry or any(label.get(k)!=value.get(k) for k in ('label','raw','seed','flip','budget_transport')):
            raise ValueError('Published labels do not select the original raw preference')
    return {'raw_api_attempts':raw_calls,'transport_retry_calls':retry_calls,'raw_semantic_slots':len(logical)}


def audit_reward(c,method,budget,seed):
    p=root(c);out=cell(c,method,budget,seed)
    verify_receipt(p/'v2.labels.audit.receipt.json')
    for path in (p/'pool.receipt.json',p/'design.receipt.json',out/'queries.receipt.json',out/'labels.receipt.json',out/'reward.receipt.json'):
        verify_receipt(path)
    queries,labels=rows(out/'queries.jsonl'),rows(out/'labels.jsonl')
    raw_audit=_audit_openai_observations(c,method,budget,seed,labels,verify_receipt(out/'labels.receipt.json'))
    with np.load(p/'design.npz',allow_pickle=False) as z:
        draws=np.random.default_rng(seed).choice(len(z[method]),size=budget,replace=True,p=z[method]);X=z['v'][draws];index=z['index']
        for k,row in enumerate(queries):
            e=int(draws[k]);i,a,b=map(int,index[e])
            if row!={'draw_id':k,'edge_id':e,'prompt_index':i,'a':a,'b':b,'probability':float(z[method][e])}:
                raise ValueError('Query did not reproduce the frozen distribution')
    if len(labels)!=budget or len(queries)!=budget:raise ValueError('Wrong label/query count')
    from .io import METHODS
    for k,value in enumerate(labels):
        if any(value.get(key)!=v for key,v in queries[k].items()):raise ValueError('Label/query mismatch')
        parts=[seed,METHODS.index(method),k,value['attempt'],991]
        if c['sampling']['budget_coupling']=='independent':parts.insert(1,budget)
        s=int(np.random.SeedSequence(parts).generate_state(1)[0]);flip=bool(np.random.default_rng(s).integers(2))
        shown={'A':1,'B':0}.get(value['raw'].strip())
        if shown is None or value['seed']!=s or value['flip']!=flip or value['label']!=(1-shown if flip else shown):
            raise ValueError('Raw OpenAI preference direction/seed mismatch')
    with np.load(out/'reward.npz',allow_pickle=False) as z:w=z['w']
    if w.shape!=(X.shape[1],) or not np.isfinite(w).all():raise ValueError('Invalid Reward head')
    y=np.asarray([v['label'] for v in labels]);scores=X@w;ridge=c['reward']['l2_regularization']
    loss=float(np.mean(np.logaddexp(0,scores)-y*scores)+ridge*(w@w)/2)
    grad=X.T@((np.tanh(scores/2)+1)/2-y)/budget+ridge*w
    report=json.loads((out/'reward.report.json').read_text(encoding='utf-8'))
    g=float(np.abs(grad).max())
    if (not np.isfinite(loss) or g>10*c['reward']['gradient_tolerance']
            or not np.isclose(loss,report['loss'],rtol=1e-10,atol=1e-12)
            or not np.isclose(g,report['gradient_inf'],rtol=1e-6,atol=1e-10)):
        raise RuntimeError('Independent BT loss/gradient audit failed')
    summary={'passed':True,'method':method,'budget':budget,'seed':seed,'bt_loss':loss,'gradient_inf':g,
             'strong_convexity_objective_gap_bound':float(grad@grad/(2*ridge)) if ridge>0 else None,'audit_api_calls':0,**raw_audit}
    dump(out/'training.audit.json',summary)
    receipt(out/'training.audit.receipt.json',[out/'training.audit.json',out/'reward.receipt.json',out/'labels.receipt.json',out/'queries.receipt.json',p/'design.receipt.json',p/'v2.labels.audit.receipt.json'])
    return summary


def gate(c,method,budget,seed,stage):
    verify_receipt(root(c)/'v2.labels.audit.receipt.json')
    out=cell(c,method,budget,seed)
    for name in ('labels','reward','training.audit'):
        verify_receipt(out/f'{name}.receipt.json')
    audit=json.loads((out/'training.audit.json').read_text(encoding='utf-8'))
    if not audit.get('passed') or (audit['method'],audit['budget'],audit['seed'])!=(method,budget,seed):
        raise ValueError('This cell has not passed its Reward audit')
    if stage=='gate-evaluation':
        verify_receipt(out/'policy.receipt.json');verify_receipt(out/'policy.audit.receipt.json')
        policy=json.loads((out/'policy.audit.json').read_text(encoding='utf-8'))
        if not policy.get('passed') or (policy['method'],policy['budget'],policy['seed'])!=(method,budget,seed):
            raise ValueError('This cell has not passed its PPO audit')
    return {'passed':True,'stage':stage,'method':method,'budget':budget,'seed':seed,'api_calls_sent':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','freeze-config','sample-export','materialize','audit-reward','audit-policy','gate-policy','gate-evaluation'])
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/experiment.v2.yaml'))
    parser.add_argument('--profile',default=str(Path(__file__).resolve().parents[2]/'configs/judges.v2.yaml'))
    parser.add_argument('--results-dir');parser.add_argument('--method');parser.add_argument('--budget',type=int);parser.add_argument('--seed',type=int)
    parser.add_argument('--asset-root');parser.add_argument('--run-root');parser.add_argument('--output-config')
    parser.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    if a.action=='freeze-config':
        if not all((a.asset_root,a.run_root,a.output_config)):parser.error('freeze-config requires --asset-root, --run-root, --output-config')
        print(json.dumps(freeze_config(a.config,a.asset_root,a.run_root,a.output_config),ensure_ascii=False,indent=2));return
    c,missing=load_config(a.config,ready=a.action!='plan')
    if a.action=='plan':result={**describe(c,a.profile),'missing_inputs':missing}
    else:
        if not a.execute:parser.error('Artifact transitions require --execute after review')
        if a.action=='sample-export':result=sample_export(c,a.profile)
        elif a.action=='materialize':
            if not a.results_dir:parser.error('materialize requires --results-dir')
            result=materialize(c,a.profile,a.results_dir)
        elif a.action=='audit-reward':
            if any(x is None for x in (a.method,a.budget,a.seed)):parser.error('audit-reward needs a cell')
            result=audit_reward(c,a.method,a.budget,a.seed)
        elif a.action=='audit-policy':
            if any(x is None for x in (a.method,a.budget,a.seed)):parser.error('audit-policy needs a cell')
            from .v2_audit import audit_policy
            result=audit_policy(c,a.method,a.budget,a.seed)
        else:
            if any(x is None for x in (a.method,a.budget,a.seed)):parser.error('gate needs a cell')
            result=gate(c,a.method,a.budget,a.seed,a.action)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
