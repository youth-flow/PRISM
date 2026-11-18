"""V4 is a new experiment contract, never a relabelled V3 run."""
from copy import deepcopy
from pathlib import Path

import yaml

from prism.v3.artifacts import config_hash
from prism.v3.config import validate as validate_shared

MODEL_ID = "allenai/Olmo-3-7B-Instruct-SFT"
MODEL_REVISION = "e1452fc572d51966ff4aaeb25118b891eb93e549"
PROTOCOL = "hh_finite_pool_v4"
RUNTIME = {"transformers": "4.57.1", "tokenizers": "0.22.1",
           "huggingface-hub": "0.36.0", "peft": "0.15.2", "accelerate": "1.7.0"}


def validate(c):
    if c.get("schema_version") != 4 or c.get("protocol") != PROTOCOL:
        raise ValueError("Expected schema 4 / hh_finite_pool_v4")
    shared = deepcopy(c)
    shared.update(schema_version=3, protocol="hh_finite_pool_v3")
    validate_shared(shared)
    if c["model"] != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("V4 pins the official Olmo-3-7B-Instruct-SFT, not its post-RL variant")
    if c["data"]["feature_dimension"] != 4096:
        raise ValueError("Olmo3 7B has 4096-dimensional EOS features")
    if c.get("runtime_packages") != RUNTIME:
        raise ValueError("Use the isolated pinned V4 dependency layer")
    p = c["ppo"]
    if (p.get("executor") != "shared_frozen_storage_v4" or p.get("gradient_checkpointing") is not True
            or p["per_device_train_batch_size"] != 1 or p["local_rollout_forward_batch_size"] != 1):
        raise ValueError("V4 uses one frozen storage, independent adapters, checkpointing and microbatch one")
    for section in (p, c["features"]):
        if type(section.get("logp_chunk_tokens")) is not int or not 1 <= section["logp_chunk_tokens"] <= 256:
            raise ValueError("Bound V4 vocabulary-projection chunks to 1..256 tokens")
    if c.get("admission") != {"policy": False, "paid_judge": False}:
        raise ValueError("Engineering configuration keeps policy and paid Judge admission closed")
    if Path(c["output"]).name != "olmo3-7b-hh-v4":
        raise ValueError("Use the separate olmo3-7b-hh-v4 output directory")
    return c


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V4 configuration must be a mapping")
    validate(raw)
    c = deepcopy(raw)
    c["output"] = str((path.parent / raw["output"]).resolve())
    c["data"]["raw_dir"] = str((path.parent / raw["data"]["raw_dir"]).resolve())
    c.update(_contract=raw, _config_path=str(path))
    c["_config_sha256"] = config_hash(c)
    return c


def require_runtime(c):
    from importlib.metadata import version
    actual = {name: version(name) for name in c["runtime_packages"]}
    if actual != c["runtime_packages"]:
        raise RuntimeError(f"V4 dependency mismatch: {actual}; use the isolated V4 environment")
    import torch
    if torch.__version__.split("+")[0] != "2.7.1" and c["runtime"]["device"] == "cuda":
        raise RuntimeError("Production V4 requires the verified Torch 2.7.1 CUDA runtime")
    return actual
