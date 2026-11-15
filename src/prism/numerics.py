"""Explicit probability-computation protocol, separate from research parameters."""
import torch


FP32_PROTOCOL = "fp32_consistent_v1"


def configure_numerics(c):
    """Leave archived configurations alone; strict runs require real FP32 math."""
    mode = c.get("runtime", {}).get("numerics")
    if mode is None:
        return None
    if mode != FP32_PROTOCOL or c["runtime"]["dtype"] != "float32":
        raise ValueError("fp32_consistent_v1 requires runtime.dtype=float32")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(protocol=mode, model_dtype="float32", logprob_dtype="float32",
                tf32=False, autocast=False)


def require_fp32_model(model):
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Probability protocol requires FP32 backbone and adapter parameters")


def selected_logprobs(logits, tokens):
    """FP32 log-softmax, bounded to one response row, preserving autograd."""
    if logits.shape[:-1] != tokens.shape:
        raise ValueError("Logits and selected token shapes differ")
    if logits.ndim == 2:
        return logits.float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    if logits.ndim != 3:
        raise ValueError("Expected [tokens,vocab] or [batch,tokens,vocab] logits")
    return torch.stack([selected_logprobs(row, ids) for row, ids in zip(logits, tokens)])


def probability_path_metrics(old, fresh, valid, *, max_logp_error=1e-3, max_approxkl=1e-7):
    """Check an unchanged policy before PPO updates, using valid actions only."""
    if old.shape != fresh.shape or valid.shape != old.shape or valid.dtype != torch.bool:
        raise ValueError("Probability-check shapes/mask differ")
    if old.dtype != torch.float32 or fresh.dtype != torch.float32:
        raise ValueError("Probability checks require FP32 log probabilities")
    delta = (fresh-old)[valid].double()
    if not delta.numel() or not torch.isfinite(delta).all():
        raise ValueError("Probability check has no finite valid actions")
    result = dict(valid_tokens=delta.numel(), max_abs_logp_error=delta.abs().max().item(),
                  zero_update_approxkl=(.5*delta.square().mean()).item(),
                  max_logp_error_tolerance=max_logp_error, approxkl_tolerance=max_approxkl)
    if result["max_abs_logp_error"] > max_logp_error or result["zero_update_approxkl"] > max_approxkl:
        raise RuntimeError(f"Unchanged-policy probability paths disagree: {result}")
    return result
