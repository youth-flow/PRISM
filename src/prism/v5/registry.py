"""Read and validate the compact, source-hashed V5 historical result registry."""
from pathlib import Path
import json
import math
import re

from .config import DATA_ID, DATA_REVISION, MODEL_ID, MODEL_REVISION


SHA256 = re.compile(r"[0-9a-f]{64}")
CLASSIFICATIONS = {
    "v5_primary", "v5_reward_candidate", "historical_linear_diagnostic",
    "out_of_scope_control",
}


def _finite(value, path="metrics"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite registry value at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
        return
    raise ValueError(f"Unsupported registry value at {path}")


def _validate_evidence(entry):
    evidence = entry.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{entry.get('id')} has no immutable evidence")
    for item in evidence:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                or not SHA256.fullmatch(str(item.get("sha256", "")))):
            raise ValueError(f"Invalid evidence in {entry.get('id')}")


def _validate_run(entry, kind):
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
        raise ValueError(f"Invalid {kind} registry entry")
    if entry.get("status") != "completed":
        raise ValueError(f"Registry contains a non-completed run: {entry['id']}")
    if entry.get("classification") not in CLASSIFICATIONS:
        raise ValueError(f"Invalid classification for {entry['id']}")
    if kind == "reward" and entry.get("classification") == "v5_primary":
        if (entry.get("architecture") != "frozen_linear"
                or entry.get("preference_pairs") != 14000
                or entry.get("method") not in ("empirical", "prism")
                or entry.get("l2") != 0.1):
            raise ValueError("V5 primary Reward must be the audited 14k frozen linear head")
    if kind == "reward" and entry.get("classification") == "v5_reward_candidate":
        if (entry.get("architecture") != "frozen_linear"
                or entry.get("preference_pairs") != 14000
                or entry.get("method") not in ("empirical", "prism")
                or entry.get("l2") not in (0.001, 0.01)
                or entry.get("selected_by_development_nll") is not False):
            raise ValueError("V5 Reward candidates must be audited non-selected 14k L2 heads")
    if kind == "ppo" and entry.get("classification") == "v5_primary":
        if (entry.get("reward_architecture") != "frozen_linear"
                or entry.get("episodes") != 16384
                or entry.get("preference_pairs") != 14000):
            raise ValueError("V5 primary PPO must be 14k frozen-linear/16k-episode")
    _finite(entry.get("metrics"), f"{entry['id']}.metrics")
    _validate_evidence(entry)


def validate_registry(value):
    if value.get("schema_version") != 1 or value.get("protocol") != "olmo1b_hh_evaluation_registry_v1":
        raise ValueError("Unknown V5 result registry schema")
    scope = value.get("scope", {})
    expected = {
        "sft_model": MODEL_ID,
        "sft_revision": MODEL_REVISION,
        "preference_data": DATA_ID,
        "preference_data_revision": DATA_REVISION,
    }
    if any(scope.get(key) != item for key, item in expected.items()):
        raise ValueError("Result registry does not match the V5 SFT/HH scope")
    rewards, policies = value.get("reward_runs"), value.get("ppo_runs")
    if not isinstance(rewards, list) or not isinstance(policies, list):
        raise ValueError("V5 registry needs reward_runs and ppo_runs")
    for entry in rewards:
        _validate_run(entry, "reward")
    for entry in policies:
        _validate_run(entry, "ppo")
    ids = [entry["id"] for entry in rewards + policies]
    if len(ids) != len(set(ids)):
        raise ValueError("V5 registry run ids must be unique")
    counts = value.get("counts", {})
    if counts != {"reward_runs": len(rewards), "ppo_runs": len(policies)}:
        raise ValueError("V5 registry count seal is stale")
    return value


def load_registry(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_registry(value)
    return value


def summary(value):
    validate_registry(value)
    all_runs = value["reward_runs"] + value["ppo_runs"]
    by_class = {name: sum(run["classification"] == name for run in all_runs)
                for name in sorted(CLASSIFICATIONS)}
    return {
        "protocol": value["protocol"],
        "counts": value["counts"],
        "by_classification": by_class,
        "primary_reward_ids": [run["id"] for run in value["reward_runs"]
                               if run["classification"] == "v5_primary"],
        "primary_ppo_ids": [run["id"] for run in value["ppo_runs"]
                            if run["classification"] == "v5_primary"],
        "historical_controls_are_not_v5_primary": True,
    }
