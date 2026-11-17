"""Small explicit frozen-model acceptance; never a formal preference experiment."""
import numpy as np

from prism.hf import generate, load_lm, prompt_ids, sequence_stats, tokenizer
from .artifacts import Run, write_json
from .config import as_hf_config


def model_smoke(c):
    run = Run(c)
    if run.completed("model-smoke"):
        from .artifacts import read_json
        return read_json(run.file("diagnostics/model-smoke.json"))
    adapter = as_hf_config(c)
    t, model = tokenizer(adapter), load_lm(adapter)
    messages = [{"role": "user", "content": "Give a brief greeting."}]
    prompt = prompt_ids(t, messages, c["data"]["max_prompt_tokens"])
    response = generate(model, t, prompt, min(8, c["ppo"]["response_length"]), c["study"]["seed"])
    feature, logp = sequence_stats(model, prompt, response, t.eos_token_id)
    if feature.shape != (c["data"]["feature_dimension"],) or not np.isfinite(feature).all() or not np.isfinite(logp):
        raise ValueError("Frozen SFT feature/log-prob smoke failed")
    report = {"protocol": "v3_frozen_model_smoke_v1", "passed": True,
              "feature_dimension": len(feature), "generated_tokens": len(response),
              "sequence_logp": logp, "dtype": str(next(model.parameters()).dtype),
              "device": str(next(model.parameters()).device), "formal_preference_observations": 0,
              "formal_ppo_episodes": 0, "paid_api_calls": 0}
    path = run.file("diagnostics/model-smoke.json")
    write_json(path, report)
    run.publish("model-smoke", [path])
    return report
