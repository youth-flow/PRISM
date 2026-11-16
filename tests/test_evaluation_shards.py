from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import pytest

from prism import evaluation_shards as shards
from prism.evaluation_v2 import _generate
from prism.io import cell,digest,dump,receipt,root,rows,verify_receipt,write_rows
from test_evaluation_v2 import config


def prepared(tmp_path):
    c=config(tmp_path);c['runtime']['numerics']='fp32_consistent_v1'
    p=root(c);ref=p/'reference';ref.mkdir();(ref/'stub').write_text('fixed SFT')
    receipt(ref/'reference.receipt.json',[ref/'stub',p/'experiment.json'])
    out=cell(c,'prism',8,11);(out/'policy.stub').write_text('fixed actor')
    receipt(out/'policy.receipt.json',[out/'policy.stub',p/'experiment.json'])
    dump(out/'policy.audit.json',{'passed':True,'method':'prism','budget':8,'seed':11})
    receipt(out/'policy.audit.receipt.json',[out/'policy.audit.json',out/'policy.receipt.json'])
    return c,out


class Actor:
    @contextmanager
    def disable_adapter(self):yield


class Tokenizer:
    eos_token_id=1
    def decode(self,tokens,skip_special_tokens):return 'test'


def fake_model(monkeypatch):
    import prism.hf as hf
    seen=[]
    monkeypatch.setattr(hf,'prompt_ids',lambda t,m,limit:[int(m[0]['content'].rsplit(' ',1)[1])+2])
    def generate(model,t,ids,cap,seed,temperature):
        seen.append((ids,cap,seed,temperature));return [2+seed%100,1]
    monkeypatch.setattr(hf,'generate',generate)
    monkeypatch.setattr(hf,'sequence_stats',lambda model,ids,response,eos:(None,float(sum(response))))
    monkeypatch.setattr(shards,'_load_actor',lambda c,out:(Actor(),Tokenizer()))
    return seen


@pytest.mark.parametrize('count',[500,541])
@pytest.mark.parametrize('parts',[1,4,7,32])
def test_shard_indices_cover_original_order_exactly(count,parts):
    groups=[range(*shards.bounds(count,index,parts)) for index in range(parts)]
    assert [i for group in groups for i in group]==list(range(count))


@pytest.mark.parametrize('values',[(5,-1,2),(5,2,2),(5,0,6),(5,0,0),(True,0,1)])
def test_invalid_shards_are_rejected(values):
    with pytest.raises(ValueError):shards.bounds(*values)


def test_shards_preserve_original_global_seed_temperature_kl_and_full_results(tmp_path,monkeypatch):
    c,out=prepared(tmp_path);frozen=digest(root(c)/'experiment.json');original=deepcopy(c);seen=fake_model(monkeypatch)
    data=shards.inputs(c);expected={}
    expected['evaluation']=_generate(c,Actor(),Tokenizer(),data['evaluation'],c['evaluation']['max_new_tokens'],1.0,kl=True)
    expected['benchmark']=_generate(c,Actor(),Tokenizer(),data['benchmark'],c['benchmark']['max_new_tokens'],c['benchmark']['temperature'],offset=2_000_000,prompt_limit=shards._benchmark_prompt_limit(c))
    baseline_calls=list(seen);seen.clear()
    # Finish the second shard first: merge must use global order, not completion order.
    shards.generate(c,'prism',8,11,1,2);shards.generate(c,'prism',8,11,0,2)
    assert sorted(seen)==sorted(baseline_calls)
    result=shards.merge(c,'prism',8,11,2)
    assert result=={'evaluation_prompts':2,'benchmark_prompts':2,'parts':2}
    for name,values in expected.items():assert rows(out/(name+'.responses.jsonl'))==values
    assert json.loads((out/'benchmark.metrics.json').read_text('utf-8'))['score']==1
    proof=verify_receipt(out/'evaluation.generate.receipt.json')
    assert proof['execution']['global_prompt_indices_and_seeds'] and len(proof['input_sha256'])>=16
    assert c==original and digest(root(c)/'experiment.json')==frozen
    with pytest.raises(FileExistsError):shards.merge(c,'prism',8,11,2)


def test_partial_or_changed_shard_cannot_publish_full_generation(tmp_path,monkeypatch):
    c,out=prepared(tmp_path);fake_model(monkeypatch);shards.generate(c,'prism',8,11,0,2)
    with pytest.raises(FileNotFoundError):shards.merge(c,'prism',8,11,2)
    assert not (out/'evaluation.generate.receipt.json').exists()
    shards.generate(c,'prism',8,11,1,2)
    path=out/'evaluation_shards/parts2/part001/evaluation.responses.jsonl';path.write_text(path.read_text('utf-8')+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Artifact changed'):shards.merge(c,'prism',8,11,2)
    assert not (out/'evaluation.responses.jsonl').exists()


@pytest.mark.parametrize('change',['coverage','seed','order','policy'])
def test_rehashed_contract_or_order_changes_are_rejected(tmp_path,monkeypatch,change):
    c,out=prepared(tmp_path);fake_model(monkeypatch)
    for index in range(2):shards.generate(c,'prism',8,11,index,2)
    folder=out/'evaluation_shards/parts2/part001';marker=folder/'shard.receipt.json'
    if change=='policy':
        proof=json.loads(marker.read_text('utf-8'));key=str(out/'policy.receipt.json');del proof['input_sha256'][key];dump(marker,proof)
    elif change=='order':
        path=folder/'evaluation.responses.jsonl';records=rows(path);records[0]['id']='wrong-index';path.unlink();write_rows(path,records)
        proof=json.loads(marker.read_text('utf-8'));proof['input_sha256'][str(path)]=digest(path);dump(marker,proof)
    else:
        path=folder/'shard.json';meta=json.loads(path.read_text('utf-8'))
        if change=='coverage':meta['ranges']['evaluation']=[0,1]
        else:meta['global_seed_rule']['evaluation_offset']=1
        dump(path,meta);proof=json.loads(marker.read_text('utf-8'));proof['input_sha256'][str(path)]=digest(path);dump(marker,proof)
    with pytest.raises(ValueError):shards.merge(c,'prism',8,11,2)
    assert not (out/'evaluation.generate.receipt.json').exists()
