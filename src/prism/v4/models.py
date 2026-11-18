"""Memory-bounded FP32 scoring and independent adapters on shared frozen storage.

Module trees and LoRA parameters are distinct. Only immutable Parameter objects
are memoized during deepcopy. No adapter merging, model quantization or role
switching is used. Sharing is valid precisely because the base never updates.
"""
from copy import deepcopy

import numpy as np


def frozen_view(backbone):
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("Freeze the complete base before creating shared-storage views")
    return deepcopy(backbone, {id(p): p for p in backbone.parameters()})


def causal_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def selected_projection(hidden, head, targets, *, chunk_size=32):
    """Exact FP32 log-softmax, recomputed per chunk during backward.

    Avoid retaining [batch, sequence, vocabulary] tensors. Summation over the
    vocabulary remains complete; neither top-k nor token probabilities change.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    from prism.numerics import selected_logprobs
    if hidden.shape[:-1] != targets.shape or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid selected-projection dimensions/chunk size")
    def project(h, t):
        return selected_logprobs(head(h).float(), t)
    result = []
    for start in range(0, targets.shape[1], chunk_size):
        h, t = hidden[:, start:start + chunk_size], targets[:, start:start + chunk_size]
        result.append(checkpoint(project, h, t, use_reentrant=False) if torch.is_grad_enabled() and h.requires_grad
                      else project(h, t))
    if not result:
        raise ValueError("Need at least one response token")
    return torch.cat(result, dim=1)


def response_logps(model, inputs, response, context, *, chunk_size=32):
    import torch
    base = causal_model(model)
    backbone = base.base_model
    # HF enables checkpointing in train mode only; all dropout is fixed at zero.
    backbone.train(torch.is_grad_enabled())
    output = backbone(**inputs)
    hidden = output.last_hidden_state[:, context - 1:context - 1 + response.shape[1]]
    return selected_projection(hidden, base.get_output_embeddings(), response, chunk_size=chunk_size)


def sequence_stats(model, prompt, response, eos, *, chunk_size=32):
    import torch
    from prism.hf import canonical_eos, check_context
    if not prompt or not response:
        raise ValueError("Nonempty prompt and response required")
    ids = canonical_eos(prompt + response, eos)
    base = causal_model(model)
    check_context(base, len(ids))
    with torch.no_grad():
        tokens = torch.tensor([ids], device=next(model.parameters()).device)
        hidden = base.base_model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                 use_cache=False, return_dict=True).last_hidden_state
        targets = torch.tensor([response], device=tokens.device)
        selected = selected_projection(hidden[:, len(prompt) - 1:len(prompt) + len(response) - 1],
                                       base.get_output_embeddings(), targets, chunk_size=chunk_size)
        return hidden[0, -1].float().cpu().numpy(), float(selected.sum())


def policy_models(lm, weights, mean, std, settings, *, eos, pad, context_tokens):
    import torch
    from prism.v3.ppo import make_actor, make_critic, FrozenNormalizedReward
    from prism.v3.ppo_trainer import partition_parameters
    lm.requires_grad_(False).eval()
    for module in lm.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    critic_base, reward_base = frozen_view(lm.base_model), frozen_view(lm.base_model)
    actor = make_actor(lm, settings)
    critic = make_critic(critic_base, weights, mean, std, settings)
    if settings.get("gradient_checkpointing", False):
        kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        causal_model(actor).gradient_checkpointing_enable(**kwargs)
        critic.backbone.gradient_checkpointing_enable(**kwargs)
    original_forward = critic.forward
    def checkpointed_value(**inputs):
        critic.backbone.train(torch.is_grad_enabled())
        return original_forward(**inputs)
    critic.forward = checkpointed_value
    reward = FrozenNormalizedReward(reward_base, weights, mean, std, eos=eos, pad=pad,
                                     context_tokens=context_tokens, batch_size=1)
    groups = partition_parameters(actor, critic)
    reward_ids = {id(p) for p in reward_base.parameters()}
    if reward_ids & {id(p) for group in groups.values() for p in group}:
        raise ValueError("Trainable parameter leaked into frozen Reward")
    for view in (actor, critic):
        shared = [p for p in view.parameters() if id(p) in reward_ids]
        if len(shared) != len(list(reward_base.parameters())) or any(p.requires_grad for p in shared):
            raise ValueError("Actor/critic must share every frozen backbone tensor with Reward")
    return actor, critic, reward


def storage_report(*models):
    parameters = [p for model in models for p in model.parameters()]
    unique = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in parameters}
    return {"logical_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
            "unique_storage_bytes": sum(unique.values()), "unique_storages": len(unique),
            "trainable_parameters": sum(p.numel() for p in {id(p): p for p in parameters}.values() if p.requires_grad)}
