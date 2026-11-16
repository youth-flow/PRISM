import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from prism import pipeline
from prism.io import cell, dump, receipt, root, write_rows
from prism.local_judge import run_packet
from prism.v2 import audit_reward, describe, freeze_config, gate, materialize, sample_export
from test_evaluation_v2 import config


def settings(c,tmp_path):
    c['budgets']=[2,4];c['sampling']['budget_coupling']='nested_prefix'
    c['judges']['training']={**c['judges']['evaluation'],'model':'gpt-6.1-sol','reasoning_effort':'high','service_tier':'flex','timeout_seconds':900}
    effective={**c['judges']['evaluation'],'reasoning_effort':c['judges']['evaluation']['reasoning_effort'],'service_tier':'flex','timeout_seconds':900}
    profile=tmp_path/'judges.yaml'
    profile.write_text(yaml.safe_dump({'judges':{**c['judges'],'evaluation':effective},
      'spending_usd':{'user_total':40,'operational_total':39,'calibration':2,'training':35.5,'evaluation':1.8}}),encoding='utf-8')
    return profile


def candidate_pool(c):
    p=root(c);rng=np.random.default_rng(61)
    n=c['data']['expected_counts']['design'];m=c['candidate']['count']
    np.savez(p/'pool.npz',phi=rng.normal(size=(n,m,3)).astype(np.float32),logp=rng.normal(size=(n,m)))
    records=[{'prompt_index':i,'candidate':j,'messages':[{'role':'user','content':f'question {i}'}],'response':f'answer {i}/{j}'} for i in range(n) for j in range(m)]
    write_rows(p/'pool.jsonl',records);receipt(p/'pool.receipt.json',[p/'pool.npz',p/'pool.jsonl',p/'experiment.json'])
    c['design']['fw_max_iter']=500
    # Fix the configuration before the immutable run is created.


def prepared(tmp_path):
    c=config(tmp_path);profile=settings(c,tmp_path);c['design']['fw_max_iter']=500
    candidate_pool(c);pipeline.design(c,solver='joint-lbfgs')
    return c,profile,sample_export(c,profile)


def fake(*args):
    return {'status':'completed','service_tier':'flex','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'A'}]}],
            'usage':{'input_tokens':9,'output_tokens':1}},'test'


def test_plan_is_metadata_only_and_counts_shared_observations(tmp_path):
    c=config(tmp_path);profile=settings(c,tmp_path)
    plan=describe(c,profile)
    assert plan['counts']['training_cells']==8 and plan['counts']['training_valid_labels']==16
    assert plan['effective_judges']['evaluation']['service_tier']=='flex'
    assert plan['api_calls_sent']==plan['slurm_jobs_submitted']==0 and len(plan['phases'])==6
    assert not Path(c['output']).exists()


def test_sample_export_and_materialization_reuse_exact_labels_without_fresh_api(tmp_path,monkeypatch):
    c,profile,prepared_data=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    assert len(prepared_data['packets'])==4
    for spec in prepared_data['packets']:
        packet=Path(spec['packet']);run_packet(packet,packet.with_name(packet.name.replace('.packet.json','.results.jsonl')),execute=True,transport=fake)
    result=materialize(c,profile,root(c)/'local_judge_packets')
    assert result['api_calls_sent']==0 and result['training_valid_observations']==16
    for m in c['methods']:
        for n in c['budgets']:
            for s in c['seeds']:
                pipeline.reward(c,m,n,s)
                assert audit_reward(c,m,n,s)['passed']
                assert gate(c,m,n,s,'gate-policy')['passed']
    out=cell(c,'prism',4,11)
    np.savez(out/'reward.npz',w=np.ones(3)*100)
    receipt(out/'reward.receipt.json',[out/'reward.npz',out/'reward.report.json',out/'labels.receipt.json'])
    with pytest.raises(RuntimeError,match='BT loss/gradient'):
        audit_reward(c,'prism',4,11)


def test_partial_max_stream_cannot_publish_any_budget_snapshot(tmp_path,monkeypatch):
    c,profile,data=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    for spec in data['packets'][:3]:
        p=Path(spec['packet']);run_packet(p,p.with_name(p.name.replace('.packet.json','.results.jsonl')),execute=True,transport=fake)
    with pytest.raises(RuntimeError,match='stream incomplete'):
        materialize(c,profile,root(c)/'local_judge_packets')
    assert not list(root(c).glob('*/n*/seed*/labels.receipt.json'))


def test_retransmitted_labels_keep_raw_http_counts_in_all_prefixes_and_audit(tmp_path,monkeypatch):
    from urllib.error import URLError
    from prism.evaluation_v2 import _training_accounting
    from prism.io import rows,verify_receipt
    c,profile,data=prepared(tmp_path);monkeypatch.setenv('OPENAI_API_KEY','TEST_KEY')
    for i,spec in enumerate(data['packets']):
        path=Path(spec['packet']);target=path.with_name(path.name.replace('.packet.json','.results.jsonl'))
        if i==0:
            def outage(*args):raise URLError('network failure')
            with pytest.raises(RuntimeError,match='exhausted'):
                run_packet(path,target,execute=True,transport=outage,network_retries=1)
        run_packet(path,target,execute=True,transport=fake,network_retries=2)
    materialize(c,profile,root(c)/'local_judge_packets')
    for method in c['methods']:
        for budget in c['budgets']:
            for seed in c['seeds']:
                pipeline.reward(c,method,budget,seed)
                audited=audit_reward(c,method,budget,seed)
                extra=4 if method==data['packets'][0]['method'] and seed==data['packets'][0]['seed'] else 0
                assert audited['raw_api_attempts']==budget+extra
                assert audited['transport_retry_calls']==(2 if extra else 0)
    assert _training_accounting(c)['unique_training_calls']==20
    spec=data['packets'][0];out=cell(c,spec['method'],4,spec['seed'])
    marker=verify_receipt(out/'labels.receipt.json');path=Path(marker['raw_api_result_path']);events=rows(path)
    events[4]['result']['flip']=not events[4]['result']['flip']
    from prism.v2 import _audit_openai_observations
    from prism.io import write_rows
    path.write_text(''.join(json.dumps(r)+'\n' for r in events),encoding='utf-8')
    with pytest.raises(ValueError,match='direction/seed/identity'):
        _audit_openai_observations(c,spec['method'],4,spec['seed'],rows(out/'labels.jsonl'),marker)


def test_private_config_freeze_is_repeatable_and_never_overwrites_other_run(tmp_path):
    from test_config_v2 import fixture_v2
    template,c=fixture_v2(tmp_path)
    template.write_text(yaml.safe_dump(c),encoding='utf-8')
    destination=tmp_path/'private/frozen.yaml'
    first=freeze_config(template,tmp_path/'assets-root',tmp_path/'run',destination)
    assert freeze_config(template,tmp_path/'assets-root',tmp_path/'run',destination)['sha256']==first['sha256']
    with pytest.raises(FileExistsError):freeze_config(template,tmp_path/'assets-root',tmp_path/'other-run',destination)
