"""Independent single-GPU evaluation shards with original global prompt seeds."""
import argparse
import json
import os
import platform
from pathlib import Path

from .benchmarks import benchmark_rows,score_benchmark,validate_benchmark_backend
from .evaluation_v2 import _aligned,_benchmark_prompt_limit,_generate,_reference,_release
from .io import cell,digest,dump,fresh,load_config,prompt_rows,receipt,root,rows,verify_receipt,write_rows


def bounds(count,index,parts):
    if any(type(v) is not int for v in (count,index,parts)) or not 1<=parts<=count or not 0<=index<parts:
        raise ValueError('Invalid evaluation shard bounds')
    return count*index//parts,count*(index+1)//parts


def inputs(c):
    return {'evaluation':prompt_rows(c['choices']['prompts']['evaluation']),'benchmark':benchmark_rows(c)}


def section(c,actor,t,data,name,start,stop):
    if name not in ('evaluation','benchmark') or not 0<=start<stop<=len(data):
        raise ValueError('Invalid evaluation section range')
    if name=='evaluation':
        return _generate(c,actor,t,data[start:stop],c['evaluation']['max_new_tokens'],1.0,offset=start,kl=True)
    return _generate(c,actor,t,data[start:stop],c['benchmark']['max_new_tokens'],c['benchmark']['temperature'],
                     offset=2_000_000+start,prompt_limit=_benchmark_prompt_limit(c))


def _load_actor(c,out):
    from peft import PeftModel
    from .hf import load_lm,tokenizer
    return PeftModel.from_pretrained(load_lm(c),str(out/'policy'),is_trainable=False).eval(),tokenizer(c)


def _prerequisites(c,method,budget,seed):
    if c['runtime'].get('dtype')!='float32' or c['runtime'].get('numerics')!='fp32_consistent_v1':
        raise ValueError('Evaluation sharding requires the accepted consistent FP32 protocol')
    out=cell(c,method,budget,seed);ref=_reference(c)
    deps=[out/'policy.receipt.json',out/'policy.audit.receipt.json',ref/'reference.receipt.json',root(c)/'experiment.json']
    for path in deps[:-1]:verify_receipt(path)
    audit=json.loads((out/'policy.audit.json').read_text(encoding='utf-8'))
    if not audit.get('passed') or (audit['method'],audit['budget'],audit['seed'])!=(method,budget,seed):
        raise ValueError('A matching independent PPO acceptance is required')
    validate_benchmark_backend(c)
    return out,deps


def _folder(out,parts,index):
    return out/'evaluation_shards'/f'parts{parts}'/f'part{index:03d}'


def generate(c,method,budget,seed,index,parts):
    out,deps=_prerequisites(c,method,budget,seed);data=inputs(c)
    ranges={name:bounds(len(values),index,parts) for name,values in data.items()}
    folder=_folder(out,parts,index)
    targets=[folder/(name+'.responses.jsonl') for name in data]
    meta_path=folder/'shard.json';marker=folder/'shard.receipt.json'
    fresh(*targets,meta_path,marker)
    # Full outputs belong exclusively to the CPU merge, never to an individual shard.
    fresh(out/'evaluation.generate.receipt.json',out/'evaluation.responses.jsonl',out/'benchmark.responses.jsonl')
    actor,t=_load_actor(c,out)
    try:
        generated={name:section(c,actor,t,values,name,*ranges[name]) for name,values in data.items()}
    finally:
        del actor
        _release(c)
    for name,values in generated.items():
        start,stop=ranges[name]
        _aligned(values,data[name][start:stop],name,require_kl=name=='evaluation')
        write_rows(folder/(name+'.responses.jsonl'),values)
    gpu=None
    if c['runtime']['device']=='cuda':
        import torch
        gpu=torch.cuda.get_device_name(torch.cuda.current_device())
    metadata={'kind':'prism_evaluation_shard','version':1,'method':method,'budget':budget,'seed':seed,
              'index':index,'parts':parts,'counts':{name:len(values) for name,values in data.items()},
              'ranges':{name:list(r) for name,r in ranges.items()},
              'global_seed_rule':{'base':c['evaluation']['seed'],'evaluation_offset':0,'benchmark_offset':2_000_000},
              'generation_settings':{'evaluation_cap':c['evaluation']['max_new_tokens'],'evaluation_temperature':1.0,
                  'benchmark_cap':c['benchmark']['max_new_tokens'],'benchmark_temperature':c['benchmark']['temperature'],
                  'benchmark_prompt_limit':_benchmark_prompt_limit(c)},
              'runtime':{'dtype':c['runtime']['dtype'],'numerics':c['runtime']['numerics'],
                         'device':c['runtime']['device'],'gpu':gpu,'node':platform.node(),'job':os.environ.get('SLURM_JOB_ID')}}
    dump(meta_path,metadata)
    receipt(marker,deps+targets+[meta_path,Path(__file__)],**metadata)
    return metadata


def merge(c,method,budget,seed,parts):
    out,deps=_prerequisites(c,method,budget,seed);data=inputs(c)
    for values in data.values():bounds(len(values),0,parts)
    artifacts=[out/name for name in ('evaluation.responses.jsonl','benchmark.responses.jsonl','benchmark.metrics.json')]
    done=out/'evaluation.generate.receipt.json';fresh(*artifacts,done)
    assembled={name:[] for name in data}
    # io.receipt keys preserve the supplied path, including legitimate aliases.
    # verify_receipt already checks the bytes reached through each path.
    fixed_hashes={str(path):digest(path) for path in deps+[Path(__file__)]}
    expected_settings={'evaluation_cap':c['evaluation']['max_new_tokens'],'evaluation_temperature':1.0,
                       'benchmark_cap':c['benchmark']['max_new_tokens'],'benchmark_temperature':c['benchmark']['temperature'],
                       'benchmark_prompt_limit':_benchmark_prompt_limit(c)}
    for index in range(parts):
        folder=_folder(out,parts,index);marker=folder/'shard.receipt.json';proof=verify_receipt(marker)
        if any(proof['input_sha256'].get(path)!=sha for path,sha in fixed_hashes.items()):
            raise ValueError('Evaluation shard source or policy dependencies differ')
        meta_path=folder/'shard.json';meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('kind')!='prism_evaluation_shard' or meta.get('version')!=1
                or (meta['method'],meta['budget'],meta['seed'],meta['index'],meta['parts'])!=(method,budget,seed,index,parts)
                or meta['counts']!={name:len(values) for name,values in data.items()}
                or meta['global_seed_rule']!={'base':c['evaluation']['seed'],'evaluation_offset':0,'benchmark_offset':2_000_000}
                or any(meta['runtime'][key]!=c['runtime'][key] for key in ('dtype','numerics','device'))
                or meta['generation_settings']!=expected_settings):
            raise ValueError('Evaluation shard contract changed')
        deps.extend([marker,meta_path])
        for name,values in data.items():
            start,stop=bounds(len(values),index,parts)
            if meta['ranges'][name]!=[start,stop]:raise ValueError('Evaluation shard coverage changed')
            path=folder/(name+'.responses.jsonl');block=rows(path)
            _aligned(block,values[start:stop],name,require_kl=name=='evaluation')
            assembled[name].extend(block);deps.append(path)
    for name,values in assembled.items():_aligned(values,data[name],name,require_kl=name=='evaluation')
    scored=score_benchmark(c,assembled['benchmark'])
    write_rows(artifacts[0],assembled['evaluation']);write_rows(artifacts[1],assembled['benchmark']);dump(artifacts[2],scored)
    receipt(done,deps+artifacts+[Path(__file__)],method=method,budget=budget,seed=seed,
            execution={'kind':'independent_single_gpu_shards','parts':parts,'global_prompt_indices_and_seeds':True})
    return {'evaluation_prompts':len(assembled['evaluation']),'benchmark_prompts':len(assembled['benchmark']),'parts':parts}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['generate','merge'])
    p.add_argument('--config',required=True);p.add_argument('--method',required=True);p.add_argument('--budget',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--parts',type=int,required=True);p.add_argument('--index',type=int)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute or not os.environ.get('SLURM_JOB_ID'):p.error('Reviewed execution requires --execute inside a Slurm allocation')
    c,_=load_config(a.config,ready=True)
    if a.action=='generate':
        if a.index is None:p.error('Generation requires --index')
        result=generate(c,a.method,a.budget,a.seed,a.index,a.parts)
    else:result=merge(c,a.method,a.budget,a.seed,a.parts)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
