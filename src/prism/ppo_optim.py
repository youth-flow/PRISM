"""Separate learning rates for the existing LoRA actor and linear critic only."""
import math


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
