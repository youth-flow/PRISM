"""Explicit, hash-bound evaluation revisions; the training contract stays fixed."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from .io import cell, digest, dump, load_config, receipt, root, rows, verify_receipt
from .judges import judge_settings


def load_profile(path, c=None):
    profile=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_profile(profile,c)
    return profile


def sample_indices(spec):
    indices=spec.get('indices')
    if (not isinstance(indices,list) or any(type(i) is not int or not 0<=i<spec['population'] for i in indices)
            or len(indices)!=spec['count'] or indices!=sorted(set(indices))):
        raise ValueError('Crosscheck requires an immutable, valid prompt-index list')
    return indices


def validate_profile(profile,c=None):
    role=profile.get('evaluation_role')
    expected={'primary':('gpt-6.1-sol','high'),'crosscheck':('gpt-6-luna','medium')}
    if (profile.get('kind')!='prism_v2_eval_profile' or profile.get('version')!=1
            or role not in expected or not profile.get('revision')):
        raise ValueError('Unknown explicit evaluation profile')
    settings=profile['judges']['evaluation'];model,effort=expected[role]
    if (settings.get('model')!=model or settings.get('reasoning_effort')!=effort
            or settings.get('backend')!='openai' or settings.get('api')!='responses'
            or settings.get('service_tier')!='flex' or settings.get('timeout_seconds')!=900):
        raise ValueError('Evaluation profile model/effort/transport changed')
    base=profile['base_evaluation_settings']
    allowed={'model','reasoning_effort','service_tier','timeout_seconds'}
    if any(base.get(k)!=settings.get(k) for k in set(base)|set(settings) if k not in allowed):
        raise ValueError('Evaluation revision may not change rubric interface, endpoint or attempt/token caps')
    auth=profile['study_authorization']
    if not auth.get('reason') or not auth.get('original_experiment_sha256'):
        raise ValueError('Explicit study authorization required')
    if role=='crosscheck':
        subset=profile['sample']
        if (type(subset['count']) is not int or type(subset['population']) is not int
                or not 0<subset['count']<=subset['population'] or type(subset['seed']) is not int):
            raise ValueError('Invalid fixed crosscheck sample')
        sample_indices(subset)
    if c is not None:
        if (digest(root(c)/'experiment.json')!=auth['original_experiment_sha256']
                or judge_settings(c,'evaluation')!=base
                or judge_settings(c,'training')!=profile['judges']['training']
                or auth['active_budgets']!=[n for n in c['budgets'] if n not in auth['cancelled_budgets']]):
            raise ValueError('Evaluation revision does not match the unchanged training experiment')
    return settings


def profile_evidence(c,path):
    path=Path(path).resolve();profile=load_profile(path,c)
    return {'path':str(path),'sha256':digest(path),'profile':profile}


def validate_evidence(packet,c=None,full_tasks=None):
    evidence=packet['evaluation_profile'];profile=evidence['profile']
    settings=validate_profile(profile,c)
    if (packet['role']!='evaluation' or packet['budget'] not in profile['study_authorization']['active_budgets']
            or settings!=packet['settings']
            or packet['input_sha256'].get(evidence['path'])!=evidence['sha256']):
        raise ValueError('Evaluation profile evidence mismatch')
    if c is not None and (digest(evidence['path'])!=evidence['sha256'] or load_profile(evidence['path'],c)!=profile):
        raise ValueError('Evaluation profile bytes changed')
    subset=packet.get('task_subset')
    if subset is not None:
        if profile['evaluation_role']!='crosscheck':raise ValueError('Only crosscheck packets can select a subset')
        spec=profile['sample'];indices=sample_indices(spec)
        if (profile['evaluation_role']!='crosscheck' or subset.get('kind')!='fixed_prompt_subset'
                or subset.get('indices')!=indices or subset.get('seed')!=spec['seed']
                or packet.get('prior_calls') or packet['input_sha256'].get(subset['parent_path'])!=subset['parent_sha256']):
            raise ValueError('Crosscheck sample changed or contains prior primary calls')
        if full_tasks is not None:
            from .local_judge import validate_packet
            parent,_=validate_packet(c,subset['parent_path'])
            if parent.get('task_subset') or parent['packet_id']!=subset['parent_packet_id'] or parent['tasks']!=full_tasks:
                raise ValueError('Crosscheck parent packet changed')
            selected=[t for t in full_tasks if int(t['id'].rsplit(':',2)[1]) in indices]
            if selected!=packet['tasks']:raise ValueError('Crosscheck requests changed')
    return settings


def export_bundle(c,method,budget,seed,primary,secondary):
    from .local_judge import export_packet,validate_packet,_hash
    out=cell(c,method,budget,seed)
    paths=[out/'evaluation.packet.json',out/'evaluation.crosscheck.full.packet.json']
    for profile_path,destination in zip((primary,secondary),paths):
        if destination.exists():validate_packet(c,destination)
        else:
            if (out/'evaluation.judge_calls.jsonl').exists():
                raise RuntimeError('Export both Judges before importing any primary judgments')
            export_packet(c,'evaluate-judge',method,budget,seed,destination,evaluation_profile=profile_path)
    parent,_=validate_packet(c,paths[1]);profile=parent['evaluation_profile']['profile'];spec=profile['sample']
    if spec['population']!=len(rows(c['choices']['prompts']['evaluation'])):raise ValueError('Crosscheck population changed')
    indices=sample_indices(spec)
    packet=deepcopy(parent);packet['tasks']=[t for t in parent['tasks'] if int(t['id'].rsplit(':',2)[1]) in indices]
    packet['task_subset']={'kind':'fixed_prompt_subset','seed':spec['seed'],'indices':indices,
                          'parent_path':str(paths[1].resolve()),'parent_sha256':digest(paths[1]),'parent_packet_id':parent['packet_id']}
    packet['input_sha256'][str(paths[1].resolve())]=digest(paths[1])
    packet['packet_id']=_hash({k:v for k,v in packet.items() if k!='packet_id'})
    destination=out/'evaluation.crosscheck.packet.json'
    if destination.exists():
        if json.loads(destination.read_text('utf-8'))!=packet:raise FileExistsError('Preserve the old crosscheck packet')
    else:dump(destination,packet)
    validate_packet(c,destination)
    return {'primary_tasks':len(validate_packet(c,paths[0])[1]),'crosscheck_tasks':len(packet['tasks']),'selected_prompt_indices':indices}


def _completed(c,packet_path,result_path):
    from .local_judge import validate_packet,_records
    packet,tasks=validate_packet(c,packet_path);_,events,accounting=_records(packet,result_path,with_accounting=True)
    complete={}
    for task in tasks:
        values=[events[(task['id'],a)] for a in range(len(task['calls'])) if (task['id'],a) in events and events[(task['id'],a)]['label'] is not None]
        if not values:raise RuntimeError('Crosscheck/primary evaluation is incomplete')
        complete[task['id']]=values[0]['label']
    return packet,complete,accounting


def audit_crosscheck(c,method,budget,seed,results):
    out=cell(c,method,budget,seed)
    packet,labels,accounting=_completed(c,out/'evaluation.crosscheck.packet.json',results)
    primary,main,_=_completed(c,out/'evaluation.packet.json',out/'evaluation.api.results.jsonl')
    if packet['evaluation_profile']['profile']['evaluation_role']!='crosscheck' or primary['evaluation_profile']['profile']['evaluation_role']!='primary':
        raise ValueError('Primary and crosscheck roles must remain separate')
    groups={}
    for tid,value in labels.items():
        name,index,order=tid.rsplit(':',2);groups.setdefault(name,{}).setdefault(int(index),{})[order]=value
    comparisons=[]
    for name,items in groups.items():
        values=[];primary_values=[];agreements=[];disagreements=[]
        for index,scores in sorted(items.items()):
            if set(scores)!=set(c['evaluation']['answer_orders']):raise ValueError('A crosscheck order is missing')
            values.append(sum(scores.values())/len(scores));primary_values.append(sum(main[f'{name}:{index}:{order}'] for order in scores)/len(scores))
            agreements.extend(scores[order]==main[f'{name}:{index}:{order}'] for order in scores)
            disagreements.append(len(set(scores.values()))>1)
        n=len(values);delta=np.asarray(values)-np.asarray(primary_values)
        ci=next(i for i,comp in enumerate(c['evaluation']['comparisons']) if comp['name']==name)
        rng=np.random.default_rng(np.random.SeedSequence([c['evaluation']['seed'],budget,seed,ci,887]))
        draws=c['evaluation'].get('bootstrap_samples',2000)
        bootstrap=delta[rng.integers(n,size=(draws,n))].mean(axis=1) if draws else None
        comparisons.append({'name':name,'prompts':n,'crosscheck_win_rate':float(np.mean(values)),
                            'primary_on_same_prompts_win_rate':float(np.mean(primary_values)),
                            'crosscheck_standard_error':float(np.std(values,ddof=1)/np.sqrt(n)) if n>1 else None,
                            'paired_score_delta':float(np.mean(delta)),
                            'paired_score_delta_ci95':np.quantile(bootstrap,[.025,.975]).tolist() if draws else None,
                            'order_label_agreement':float(np.mean(agreements)),
                            'crosscheck_position_disagreement_rate':float(np.mean(disagreements))})
    summary={'passed':True,'method':method,'budget':budget,'seed':seed,'valid_tasks':len(labels),**accounting,
             'selected_prompt_indices':packet['task_subset']['indices'],'comparisons':comparisons,
             'primary_model':primary['settings']['model'],'crosscheck_model':packet['settings']['model'],
             'human_ground_truth':False,'scope':'Secondary fixed-subset sensitivity only; primary metrics are unchanged.'}
    dump(out/'evaluation.crosscheck.metrics.json',summary)
    receipt(out/'evaluation.crosscheck.receipt.json',[out/'evaluation.crosscheck.metrics.json',out/'evaluation.crosscheck.packet.json',
            Path(results),out/'evaluation.packet.json',out/'evaluation.api.results.jsonl',root(c)/'experiment.json',
            Path(packet['task_subset']['parent_path']),Path(packet['evaluation_profile']['path']),
            Path(primary['evaluation_profile']['path']),Path(__file__)])
    return summary


def aggregate_crosscheck(c,budgets,scope_path):
    from .evaluation_v2 import _csv
    target=root(c)/'summaries'/('budgets-'+'-'.join(map(str,sorted(budgets))))
    primary=verify_receipt(target/'aggregation.receipt.json')
    if primary.get('active_budgets')!=sorted(budgets):raise ValueError('Primary aggregation scope changed')
    dependencies=[target/'aggregation.receipt.json',Path(scope_path)];comparisons=[];valid=actual=0
    for method in c['methods']:
        for budget in sorted(budgets):
            for seed in c['seeds']:
                out=cell(c,method,budget,seed);marker=out/'evaluation.crosscheck.receipt.json';verify_receipt(marker)
                data=json.loads((out/'evaluation.crosscheck.metrics.json').read_text(encoding='utf-8'))
                if not data['passed']:raise ValueError('A secondary audit did not pass')
                valid+=data['valid_tasks'];actual+=data['actual_judge_calls'];dependencies.append(marker)
                comparisons.extend({'method':method,'budget':budget,'seed':seed,**row} for row in data['comparisons'])
    _csv(target/'crosscheck.luna.csv',comparisons)
    summary={'complete':True,'active_budgets':sorted(budgets),'valid_tasks':valid,'actual_judge_calls':actual,
             'primary_model':'gpt-6.1-sol','crosscheck_model':'gpt-6-luna','human_ground_truth':False,'comparisons':comparisons}
    dump(target/'crosscheck.luna.json',summary)
    receipt(target/'crosscheck.luna.receipt.json',dependencies+[target/'crosscheck.luna.json',target/'crosscheck.luna.csv'])
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['export','crosscheck','aggregate-crosscheck'])
    p.add_argument('--config',required=True);p.add_argument('--method');p.add_argument('--budget',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--active-budgets',nargs='+',type=int);p.add_argument('--study-scope')
    p.add_argument('--primary');p.add_argument('--secondary');p.add_argument('--results');a=p.parse_args();c,_=load_config(a.config,ready=True)
    if a.action=='aggregate-crosscheck':
        if not a.active_budgets or not a.study_scope:p.error('Crosscheck aggregation requires explicit scope')
        print(json.dumps(aggregate_crosscheck(c,a.active_budgets,a.study_scope),ensure_ascii=False));return
    if any(x is None for x in (a.method,a.budget,a.seed)):p.error('Cell method/budget/seed required')
    if a.action=='export':
        if not a.primary or not a.secondary:p.error('Export requires both explicit profiles')
        result=export_bundle(c,a.method,a.budget,a.seed,a.primary,a.secondary)
    else:
        if not a.results:p.error('Crosscheck requires original raw results')
        result=audit_crosscheck(c,a.method,a.budget,a.seed,a.results)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
