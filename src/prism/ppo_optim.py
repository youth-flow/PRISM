"""Separate learning rates for the existing LoRA actor and linear critic only."""
import math


def configure_rollout_accumulation(accelerator):
    """PPO microbatches, rather than prompt-loader boundaries, define updates."""
    accelerator.gradient_state.plugin_kwargs['sync_with_dataloader'] = False
    if accelerator.gradient_state.sync_with_dataloader:
        raise RuntimeError('PPO gradient accumulation must ignore prompt-loader exhaustion')


def optimizer_step_report(optimizer):
    """Read real AdamW state counters after training; do not alter the loop."""
    result={}
    for group in optimizer.param_groups:
        role=group.get('prism_role')
        if role not in ('actor','critic'):raise ValueError('Optimizer role metadata missing')
        for parameter in group['params']:
            value=optimizer.state.get(parameter,{}).get('step',0)
            step=int(value.item() if hasattr(value,'item') else value)
            item=result.setdefault(role,{'tensor_count':0,'step_min':step,'step_max':step})
            item['tensor_count']+=1;item['step_min']=min(item['step_min'],step);item['step_max']=max(item['step_max'],step)
    if set(result)!= {'actor','critic'}:raise ValueError('Expected actor and critic optimizer state')
    return result


def separate_critic_learning_rate(optimizer, model, learning_rate):
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("critic_learning_rate must be finite and positive")
    if optimizer.state:
        raise ValueError("Partition optimizer groups before the first optimizer step")
    roles = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("policy.") and "lora_" in name:
            role = "actor"
        elif name.startswith("value_model.score."):
            role = "critic"
        else:
            raise ValueError(f"Unexpected trainable PPO parameter: {name}")
        roles[id(parameter)] = role
    if set(roles.values()) != {"actor", "critic"}:
        raise ValueError("Both actor and critic parameters are required")
    groups, seen = [], []
    for original in optimizer.param_groups:
        for role in ("actor", "critic"):
            parameters = [p for p in original["params"] if roles.get(id(p)) == role]
            if parameters:
                group = {k: v for k, v in original.items() if k != "params"}
                group.update(params=parameters, prism_role=role)
                if role == "critic":
                    group["lr"] = float(learning_rate)
                groups.append(group)
                seen.extend(map(id, parameters))
    original_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(seen) != len(set(seen)) or set(seen) != set(roles) or set(seen) != set(original_ids):
        raise ValueError("Optimizer partition must cover each trainable parameter exactly once")
    optimizer.param_groups = []
    for group in groups:
        optimizer.add_param_group(group)
    return [{"role": g["prism_role"], "initial_lr": g["lr"],
             "parameters": sum(p.numel() for p in g["params"]),
             "weight_decay": g["weight_decay"]} for g in optimizer.param_groups]
