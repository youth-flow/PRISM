from copy import deepcopy
from pathlib import Path
import json

import pytest
import yaml

from prism import evaluation_v2
from prism.evaluation_protocol import export_bundle,audit_crosscheck
from prism.io import cell,digest,root,verify_receipt
from prism.local_judge import run_packet,import_results,validate_packet
from test_evaluation_v2 import config,save_generations


def profiles(c,tmp_path):
    paths=[]
    for role,model,effort in [('primary','gpt-6.1-sol','high'),('crosscheck','gpt-6-luna','medium')]:
        value={'kind':'prism_v2_eval_profile','version':1,'revision':'test-v2-revision','evaluation_role':role,
               'base_evaluation_settings':c['judges']['evaluation'],
               'judges':{'training':c['judges']['training'],'evaluation':{**c['judges']['evaluation'],'model':model,'reasoning_effort':effort,'service_tier':'flex','timeout_seconds':900}},
               'study_authorization':{'original_experiment_sha256':digest(root(c)/'experiment.json'),
                   'active_budgets':c['budgets'],'cancelled_budgets':[],'reason':'Explicit test authorization'}}
        if role=='crosscheck':value['sample']={'population':2,'count':1,'seed':20261116,'indices':[0]}
        path=tmp_path/(role+'.yaml');path.write_text(yaml.safe_dump(value),encoding='utf-8');paths.append(path)
    return paths


def fake_response(url,payload,key,timeout):
    assert payload['model'] in ('gpt-6.1-sol','gpt-6-luna')
    return {'status':'completed','service_tier':'flex','output':[{'type':'message','role':'assistant',
        'content':[{'type':'output_text','text':'TIE'}]}],'usage':{'input_tokens':12,'output_tokens':1}},'fake-request'


def test_primary_revision_and_fixed_crosscheck_preserve_training_and_separate_metrics(tmp_path,monkeypatch):
    c=config(tmp_path);save_generations(c);frozen=digest(root(c)/'experiment.json');original=deepcopy(c)
    primary,secondary=profiles(c,tmp_path);out=cell(c,'prism',8,11)
    exported=export_bundle(c,'prism',8,11,primary,secondary)
    assert exported['primary_tasks']==8 and exported['crosscheck_tasks']==4
    monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    with pytest.raises(ValueError,match='parent is not executable'):
        run_packet(out/'evaluation.crosscheck.full.packet.json',tmp_path/'forbidden.jsonl',execute=True,transport=lambda *a:pytest.fail('No HTTP'))
    main_raw=out/'evaluation.api.results.jsonl';cross_raw=out/'evaluation.crosscheck.api.results.jsonl'
    run_packet(out/'evaluation.packet.json',main_raw,execute=True,transport=fake_response)
    main=import_results(c,out/'evaluation.packet.json',main_raw)
    assert main['win_rate_vs_sft']==.5
    before=digest(out/'metrics.json')
    run_packet(out/'evaluation.crosscheck.packet.json',cross_raw,execute=True,transport=fake_response)
    with pytest.raises(ValueError,match='Secondary results cannot overwrite'):
        import_results(c,out/'evaluation.crosscheck.packet.json',cross_raw)
    checked=audit_crosscheck(c,'prism',8,11,cross_raw)
    assert checked['valid_tasks']==4 and all(r['order_label_agreement']==1 for r in checked['comparisons'])
    assert digest(out/'metrics.json')==before and digest(root(c)/'experiment.json')==frozen and c==original
    metadata=verify_receipt(out/'evaluation.receipt.json')
    assert metadata['effective_judge_settings']['model']=='gpt-6.1-sol'
    assert str(main_raw.resolve()) in metadata['input_sha256']
    verify_receipt(out/'evaluation.crosscheck.receipt.json')
    assert export_bundle(c,'prism',8,11,primary,secondary)==exported


@pytest.mark.parametrize('field,value',[('max_attempts',4),('max_output_tokens',2000),('base_url','https://example.test/v1'),('temperature',1)])
def test_evaluation_revision_does_not_change_other_scientific_settings(tmp_path,field,value):
    c=config(tmp_path);save_generations(c);primary,secondary=profiles(c,tmp_path)
    data=yaml.safe_load(primary.read_text(encoding='utf-8'));data['judges']['evaluation'][field]=value;primary.write_text(yaml.safe_dump(data),encoding='utf-8')
    with pytest.raises(ValueError,match='may not change'):
        export_bundle(c,'prism',8,11,primary,secondary)


def test_wrong_model_raw_response_cannot_be_relabelled_as_primary(tmp_path,monkeypatch):
    c=config(tmp_path);save_generations(c);primary,secondary=profiles(c,tmp_path);export_bundle(c,'prism',8,11,primary,secondary)
    out=cell(c,'prism',8,11);raw=out/'evaluation.api.results.jsonl';monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    run_packet(out/'evaluation.packet.json',raw,execute=True,transport=fake_response)
    events=[json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()];events[0]['result']['configured_model']='gpt-6-luna'
    raw.write_text(''.join(json.dumps(row)+'\n' for row in events),encoding='utf-8')
    with pytest.raises(ValueError,match='different model'):import_results(c,out/'evaluation.packet.json',raw)


def test_crosscheck_seed_or_profile_mutation_is_rejected(tmp_path):
    from prism.local_judge import _hash
    c=config(tmp_path);save_generations(c);primary,secondary=profiles(c,tmp_path);export_bundle(c,'prism',8,11,primary,secondary)
    path=cell(c,'prism',8,11)/'evaluation.crosscheck.packet.json';value=json.loads(path.read_text(encoding='utf-8'))
    value['task_subset']['seed']+=1;value['packet_id']=_hash({k:v for k,v in value.items() if k!='packet_id'})
    path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='sample changed'):validate_packet(c,path)


def test_cancelled_budget_is_complete_in_authorized_scope_without_altering_original(tmp_path,monkeypatch):
    from prism.io import dump
    from test_evaluation_v2 import install_judge
    c=config(tmp_path);c['budgets']=[4,8];save_generations(c);install_judge(monkeypatch,fail_first=False)
    for method in c['methods']:
        for seed in c['seeds']:evaluation_v2.evaluate_judge(c,method,4,seed)
    frozen=digest(root(c)/'experiment.json');scope=tmp_path/'scope.json'
    value={'kind':'prism_study_scope','version':1,'experiment_sha256':frozen,'requested_budgets':[4,8],
           'active_budgets':[4],'deferred_budgets':[],'cancelled_budgets':[8],'reason':'User cancels large budget'}
    dump(scope,value);result=evaluation_v2.aggregate(c,budgets=[4],scope_path=scope);summary=json.loads(Path(result['output']).read_text(encoding='utf-8'))
    assert summary['authorized_scope_complete'] and summary['cancelled_budgets']==[8] and summary['deferred_budgets']==[]
    assert not summary['full_frozen_matrix_complete'] and digest(root(c)/'experiment.json')==frozen
    value['deferred_budgets']=[8];dump(scope,value)
    with pytest.raises(ValueError,match='scope does not match'):evaluation_v2.aggregate(c,budgets=[4],scope_path=scope)


def test_revised_budgeted_primary_uses_existing_ledger_and_streaming_model_binding(tmp_path,monkeypatch):
    from prism.api_budget import USDBudget,DispatchPaused
    from prism.budgeted_judge import prepare_manifest,run_manifest,inspect_manifest
    c=config(tmp_path);save_generations(c);primary,secondary=profiles(c,tmp_path)
    value=yaml.safe_load(primary.read_text(encoding='utf-8'))
    value.update(spending_usd={'user_total':54,'operational_total':54,'calibration':2,'training':31,'evaluation':21.5},
                 spending_authorization={'kind':'prism_v2_evaluation_revision','version':1,'cancelled_budgets':[8192],'reason':'Explicit user selection'})
    primary.write_text(yaml.safe_dump(value),encoding='utf-8');export_bundle(c,'prism',8,11,primary,secondary)
    out=cell(c,'prism',8,11);ledger=tmp_path/'budget.sqlite'
    b=USDBudget(ledger,total_usd=54,purposes={'calibration':3,'training':46,'evaluation':5})
    b.reserve('old-training','training','gpt-6.1-sol',{'model':'gpt-6.1-sol','max_output_tokens':100,'input':[]});b.uncertain('old-training')
    prior=b.snapshot()['training'];b.control(True,'Prepare evaluation');b.control_purpose('training',True,'Cancelled tier')
    b.authorize_limits(total_usd=54,purposes={'calibration':2,'training':31,'evaluation':21.5},reason='Approved main and secondary evaluation')
    manifest=tmp_path/'manifest.json';prepare_manifest([out/'evaluation.packet.json'],primary,ledger,manifest,workers=2)
    assert inspect_manifest(manifest)['http_requests_sent']==0
    monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    with pytest.raises(DispatchPaused):run_manifest(manifest,execute=True,transport=lambda *a:pytest.fail('Paused'))
    b.control(False,'Evaluation only');sent=[]
    def streamed(url,payload,key,timeout):
        assert payload['stream'] is True and payload['model']=='gpt-6.1-sol';sent.append(True)
        return fake_response(url,payload,key,timeout)
    assert run_manifest(manifest,execute=True,transport=streamed,poll_seconds=.01,stream_responses=True)['state']=='completed'
    assert len(sent)==8 and b.snapshot()['training']==prior
    import_results(c,out/'evaluation.packet.json',out/'evaluation.results.jsonl')


def test_crosscheck_aggregation_keeps_primary_scope_and_counts_actual_calls(tmp_path,monkeypatch):
    from prism.evaluation_protocol import aggregate_crosscheck
    from prism.io import dump
    c=config(tmp_path);c['budgets']=[4,8];save_generations(c);primary,secondary=profiles(c,tmp_path)
    for path in (primary,secondary):
        value=yaml.safe_load(path.read_text(encoding='utf-8'));value['study_authorization'].update(active_budgets=[4],cancelled_budgets=[8])
        path.write_text(yaml.safe_dump(value),encoding='utf-8')
    monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    for method in c['methods']:
        for seed in c['seeds']:
            export_bundle(c,method,4,seed,primary,secondary);out=cell(c,method,4,seed)
            main=out/'evaluation.api.results.jsonl';cross=out/'evaluation.crosscheck.api.results.jsonl'
            run_packet(out/'evaluation.packet.json',main,execute=True,transport=fake_response)
            run_packet(out/'evaluation.crosscheck.packet.json',cross,execute=True,transport=fake_response)
            import_results(c,out/'evaluation.packet.json',main);audit_crosscheck(c,method,4,seed,cross)
    scope=tmp_path/'scope.json';dump(scope,{'kind':'prism_study_scope','version':1,
        'experiment_sha256':digest(root(c)/'experiment.json'),'requested_budgets':[4,8],
        'active_budgets':[4],'deferred_budgets':[],'cancelled_budgets':[8],'reason':'Cancelled'})
    evaluation_v2.aggregate(c,budgets=[4],scope_path=scope)
    result=aggregate_crosscheck(c,[4],scope)
    assert result['complete'] and result['valid_tasks']==12 and result['actual_judge_calls']==12 and len(result['comparisons'])==6
    verify_receipt(root(c)/'summaries/budgets-4/crosscheck.luna.receipt.json')
