"""Import fixed candidate tokens with previously validated FP32 scoring.

This is a finite-candidate study: text generation retains its documented v1
origin. Geometry is recomputed and certified; no old labels or policies enter.
Run on a scheduled CPU node with the study's pinned numerical environment.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import hashlib
import shutil

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .design import edges, empirical, objective_gradient, proxy
from .io import digest, dump, fresh, load_config, receipt, root, rows, verify_receipt


def import_fixed_candidates(c, original, pool_checks, geometry_check, independent_check):
    if c['runtime'].get('numerics') != 'fp32_consistent_v1':
        raise ValueError('Fixed FP32 candidates require the explicit numerical protocol')
    original=Path(original);geometry_check=Path(geometry_check);independent_check=Path(independent_check)
    p=root(c)
    fresh(*[p/name for name in ('pool.npz','pool.jsonl','pool.receipt.json','design.npz','design.report.json','design.receipt.json','candidate.provenance.json')])
    verify_receipt(original/'pool.receipt.json')
    old_config=json.loads((original/'experiment.json').read_text(encoding='utf-8'))['config']
    if (c['choices']['sft'] != old_config['choices']['sft'] or c['candidate'] != old_config['candidate']
            or digest(c['choices']['prompts']['design']) != digest(old_config['choices']['prompts']['design'])):
        raise ValueError('Fixed candidate model/prompt/sampling contract changed')
    for name in ('surrogate_ridge','matrix_ridge','fw_relative_gap'):
        if c['design'][name] != old_config['design'][name]:
            raise ValueError('Fixed candidate geometry/ridge contract changed')
    records=rows(original/'pool.jsonl')
    prompts=c['data']['expected_counts']['design'];count=c['candidate']['count'];n=prompts*count
    if len(records)!=n:
        raise ValueError('Fixed candidate count mismatch')
    for k,row in enumerate(records):
        if row['prompt_index']*count+row['candidate']!=k:
            raise ValueError('Fixed candidate order/index mismatch')
    phi=np.empty((n,2048),dtype=np.float32);logp=np.empty(n,dtype=np.float64);seen=[]
    dependencies=[original/'pool.receipt.json',original/'pool.jsonl',original/'experiment.json']
    for folder in map(Path,pool_checks):
        verify_receipt(folder/'receipt.json');dependencies.append(folder/'receipt.json')
        with np.load(folder/'pool.npz',allow_pickle=False) as z:
            ids=z['indices'];phi[ids]=z['phi'];logp[ids]=z['logp'];seen.extend(ids.tolist())
        dependencies.append(folder/'pool.npz')
    if sorted(seen)!=list(range(n)) or not np.isfinite(phi).all() or not np.isfinite(logp).all():
        raise ValueError('FP32 pool must have complete, unique, finite coverage')
    phi=phi.reshape(prompts,count,2048);logp=logp.reshape(prompts,count)
    verify_receipt(geometry_check/'receipt.json');verify_receipt(independent_check/'receipt.json')
    independent=json.loads((independent_check/'audit.json').read_text(encoding='utf-8'))
    if not independent.get('passed') or independent['relative_gap']>c['design']['fw_relative_gap']:
        raise ValueError('Previously independent geometry certificate is not valid')
    dependencies.extend([geometry_check/'receipt.json',geometry_check/'geometry.npz',independent_check/'receipt.json',independent_check/'audit.json'])
    with np.load(geometry_check/'geometry.npz',allow_pickle=False) as z:
        q=z['prism'].copy();previous_b=z['b'];previous_K=z['K']
    serialized=BytesIO();np.save(serialized,q,allow_pickle=False)
    if hashlib.sha256(serialized.getvalue()).hexdigest()!='a875848b1d380d2817c401df28ce9e7c9550f0d1018fc439822008399747fd13':
        raise ValueError('Validated fixed-candidate design weights changed')
    b,probabilities,K=proxy(phi.astype(np.float64),logp,c['design']['surrogate_ridge'])
    if (np.linalg.norm(b-previous_b)/np.linalg.norm(previous_b)>1e-8
            or np.linalg.norm(K-previous_K)/np.linalg.norm(previous_K)>1e-8):
        raise ValueError('Recomputed geometry does not match its FP32 source')
    index,v=edges(phi.astype(np.float64));q0=empirical(index)
    if q.shape!=q0.shape or np.min(q)<0 or not np.isclose(q.sum(),1,atol=1e-12,rtol=0):
        raise ValueError('Invalid design probability vector')
    value,gradient=objective_gradient(v,K,q,c['design']['matrix_ridge'])
    gap=float(q@gradient-gradient.min())
    # Independent LU solve and direct sensitivities, separate from the native
    # Cholesky objective/gradient path. No optimizer success flag is a gate.
    A=c['design']['matrix_ridge']*np.eye(2048)+(v.T*q)@v
    lu=lu_factor(A)
    independent_value=float(np.trace(lu_solve(lu,K)))
    sensitivities=[]
    for start in range(0,len(v),1024):
        h=lu_solve(lu,v[start:start+1024].T)
        sensitivities.extend(-np.einsum('ij,ij->j',h,K@h))
    sensitivities=np.asarray(sensitivities)
    independent_gap=float(q@sensitivities-sensitivities.min())
    metrics=np.asarray([value,gap,independent_value,independent_gap])
    if (not np.isfinite(metrics).all() or min(value,independent_value)<=0
            or not np.isclose(value,independent_value,rtol=1e-8,atol=1e-10)
            or max(gap/value,independent_gap/independent_value)>c['design']['fw_relative_gap']):
        raise RuntimeError('Original q-space FW gap rejected imported geometry')
    np.savez_compressed(p/'pool.npz',phi=phi,logp=logp)
    shutil.copyfile(original/'pool.jsonl',p/'pool.jsonl')
    provenance=dict(candidate_tokens_reused=True,candidate_generation_dtype=old_config['runtime']['dtype'],
                    scoring_dtype='float32',candidate_source_sha256=digest(original/'pool.jsonl'),
                    original_generation_is_not_claimed_fp32=True,source_numerical_checks=list(map(str,pool_checks)),
                    objective=value,fw_gap=gap,independent_lu_objective=independent_value,independent_lu_gap=independent_gap,
                    imported_q_npy_sha256=hashlib.sha256(serialized.getvalue()).hexdigest())
    dump(p/'candidate.provenance.json',provenance)
    receipt(p/'pool.receipt.json',[p/name for name in ('pool.npz','pool.jsonl','candidate.provenance.json','experiment.json')]+dependencies,
            feature='frozen_sft_final_layer_eos',sequence_logp='all_sampled_response_tokens_including_sampled_eos',
            feature_only_eos_append=True,candidate_tokens_reused=True,scoring_dtype='float32')
    np.savez_compressed(p/'design.npz',index=index,v=v,b=b,proxy_probabilities=probabilities,K=K,empirical=q0,prism=q)
    dump(p/'design.report.json',{'prism':dict(converged=True,objective=value,fw_gap=gap,
         relative_gap_tolerance=c['design']['fw_relative_gap'],solver='previously_solved_fixed_geometry_recertified',
         independent_lu_relative_gap=independent_gap/independent_value)})
    receipt(p/'design.receipt.json',[p/name for name in ('design.npz','design.report.json','pool.receipt.json')]+dependencies,
            solver='validated_geometry_reuse',geometry_recomputed=True)
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--original',required=True)
    parser.add_argument('--pool-checks',nargs=4,required=True)
    parser.add_argument('--geometry-check',required=True)
    parser.add_argument('--independent-check',required=True)
    a=parser.parse_args();c,missing=load_config(a.config,ready=True);assert not missing
    print(json.dumps(import_fixed_candidates(c,a.original,a.pool_checks,a.geometry_check,a.independent_check),indent=2))


if __name__=='__main__':main()
