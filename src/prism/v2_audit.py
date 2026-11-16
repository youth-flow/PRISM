"""CPU acceptance of completed V2 PPO artifacts, without API or generation."""
import json
import math

from .io import cell, dump, receipt, rows, verify_receipt


def validate_policy_report(c,report):
    p=c['ppo']
    batch=p['per_device_train_batch_size']*p['gradient_accumulation_steps']*p['num_mini_batches']
    updates=p['total_episodes']//batch;steps=updates*p['num_ppo_epochs']*p['num_mini_batches']
    expected={'total_episodes':p['total_episodes'],'rollout_batch_size':batch,'rollout_updates':updates,
              'ppo_epochs_per_rollout':p['num_ppo_epochs'],'mini_batches_per_epoch':p['num_mini_batches'],
              'critic':'frozen_sft_plus_linear_head','reference':'actor_base_with_lora_disabled'}
    if any(report[k]!=v for k,v in expected.items()):raise ValueError('PPO schedule/model contract changed')
    if (report['numerics']['protocol']!='fp32_consistent_v1' or report['numerics']['tf32']
            or report['numerics']['autocast'] or report['gradient_norm_clipping_executed']):
        raise ValueError('PPO numerical protocol changed')
    for role in ('actor','critic'):
        state=report['actual_optimizer_steps'][role]
        if state['step_min']!=steps or state['step_max']!=steps or state['tensor_count']<=0:
            raise ValueError('Actual AdamW step counters mismatch')
    checks=report['probability_checks']
    if len(checks)!=updates or [x['rollout_update'] for x in checks]!=list(range(1,updates+1)):
        raise ValueError('A PPO probability check is missing')
    for value in checks:
        if (value['max_abs_logp_error']>1e-3 or value['zero_update_approxkl']>1e-7
                or not math.isfinite(value['max_abs_logp_error']) or not math.isfinite(value['zero_update_approxkl'])):
            raise ValueError('PPO probability path did not pass')
    history=[r for r in report['log_history'] if 'episode' in r]
    if len(history)!=updates or [r['episode'] for r in history]!=list(range(batch,p['total_episodes']+1,batch)):
        raise ValueError('PPO training history is incomplete')
    for i,r in enumerate(history):
        for key in ('objective/kl','objective/scores','objective/rlhf_reward','loss/policy_avg','loss/value_avg','policy/approxkl_avg'):
            if not math.isfinite(r[key]):raise ValueError('Nonfinite PPO diagnostic')
        if not math.isclose(r['lr'],p['learning_rate']*(updates-i)/updates,rel_tol=1e-9,abs_tol=1e-15):
            raise ValueError('Actor learning-rate schedule mismatch')
    initial=report['optimizer_groups']
    for role,lr in (('actor',p['learning_rate']),('critic',p['critic_learning_rate'])):
        groups=[g for g in initial if g['role']==role]
        if not groups or any(not math.isclose(g['initial_lr'],lr,rel_tol=1e-12) for g in groups):
            raise ValueError('Actor/critic learning rates do not match the configuration')
    return {'rollout_updates':updates,'actual_adamw_steps':steps,'probability_checks':len(checks)}


def audit_policy(c,method,budget,seed):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    out=cell(c,method,budget,seed);p=c['ppo']
    for name in ('training.audit.receipt.json','policy.receipt.json'):
        verify_receipt(out/name)
    if not json.loads((out/'training.audit.json').read_text(encoding='utf-8'))['passed']:
        raise ValueError('Reward audit is not accepted')
    report=json.loads((out/'policy.report.json').read_text(encoding='utf-8'))
    counters=validate_policy_report(c,report)
    base=AutoConfig.from_pretrained(c['choices']['sft']['model'],revision=c['choices']['sft']['revision'],local_files_only=True,trust_remote_code=False)
    adapter=json.loads((out/'policy/adapter_config.json').read_text(encoding='utf-8'))
    if (adapter['r']!=p['lora_r'] or adapter['lora_alpha']!=p['lora_alpha'] or adapter['bias']!='none'
            or adapter['base_model_name_or_path']!=c['choices']['sft']['model']):
        raise ValueError('Actor adapter contract mismatch')
    target=out/'policy/adapter_model.safetensors';tensors=0;b_norm=0.
    with safe_open(target,framework='pt',device='cpu') as weights:
        for name in weights.keys():
            value=weights.get_tensor(name)
            if value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('Nonfinite/non-FP32 adapter')
            if 'lora_B' in name:b_norm+=float(value.square().sum())
            tensors+=1
    if tensors!=base.num_hidden_layers*7*2 or b_norm<=0:raise ValueError('Incomplete or unchanged OLMo LoRA adapter')
    summary={'passed':True,'method':method,'budget':budget,'seed':seed,'total_episodes':p['total_episodes'],
             **counters,'lora_tensors':tensors,'audit_api_calls':0}
    dump(out/'policy.audit.json',summary)
    receipt(out/'policy.audit.receipt.json',[out/'policy.audit.json',out/'policy.receipt.json',out/'training.audit.receipt.json'])
    return summary
