"""Hash-verified import of immutable upstream V3 assets into an isolated V5 run."""
import os
from pathlib import Path
import shutil

from prism.io import digest
from prism.v3.artifacts import Run, read_json, write_json
from prism.v3.config import load_config


UPSTREAM_STAGES = (
    "data", "features", "geometry", "geometry-audit",
    "design-0.001", "design-0.01", "design-0.1", "design-report", "shared-sft",
)
COMPATIBLE_SECTIONS = (
    "model", "runtime", "study", "data", "features", "design", "sampling",
    "reward", "ppo", "evaluation",
)


def _materialize(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or digest(target) != digest(source):
            raise ValueError(f"Preserve different V5 upstream artifact: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if digest(target) != digest(source):
        target.unlink(missing_ok=True)
        raise ValueError(f"Imported V5 artifact hash differs: {target}")


def import_upstream(c, upstream_config):
    """Import only label-independent assets; Reward and all 8k outputs remain fresh."""
    contract = c.get("v5", {})
    if (contract.get("study_variant") != "budget_n8000"
            or contract.get("preference_budget") != 8000):
        raise ValueError("Upstream import is restricted to the isolated V5 8k study")
    upstream = load_config(upstream_config)
    mismatches = [name for name in COMPATIBLE_SECTIONS
                  if c["_contract"][name] != upstream["_contract"][name]]
    if mismatches:
        raise ValueError("V5/upstream scientific sections differ: " + str(mismatches))
    source_run, target_run = Run(upstream, create=False), Run(c)
    if source_run.path == target_run.path:
        raise ValueError("V5 import needs a distinct output directory")
    imported = {}
    for stage in UPSTREAM_STAGES:
        source_receipt = source_run.receipt_path(stage)
        receipt = source_run.require(stage)
        outputs = []
        for key, expected in receipt["outputs"].items():
            if not key.startswith("run:"):
                raise ValueError(f"Upstream stage has a non-run output: {stage}/{key}")
            source = source_run.file(key[4:])
            if digest(source) != expected:
                raise ValueError(f"Upstream output changed during import: {source}")
            target = target_run.file(key[4:])
            _materialize(source, target)
            outputs.append(target)
        if target_run.completed(stage):
            current = target_run.require(stage)
            if current["outputs"] != {target_run._key(path): digest(path) for path in outputs}:
                raise ValueError(f"Existing V5 imported stage differs: {stage}")
        else:
            target_run.publish(stage, outputs, [source_receipt], metadata={
                "import_protocol": "v5_hash_verified_upstream_v1",
                "upstream_config_sha256": source_run.fingerprint,
                "upstream_receipt_sha256": digest(source_receipt),
            })
        imported[stage] = {
            "files": len(outputs), "source_receipt_sha256": digest(source_receipt),
            "target_receipt_sha256": digest(target_run.receipt_path(stage)),
        }
    summary = {
        "protocol": "v5_hash_verified_upstream_v1",
        "upstream_run": str(source_run.path),
        "target_run": str(target_run.path),
        "stages": imported,
        "fresh_downstream_stages": [
            "study-selection", "sampling", "reward", "training-audit", "normalization",
            "linear-ppo", "evaluation",
        ],
    }
    target = target_run.file("control/upstream-import.json")
    write_json(target, summary)
    if not target_run.completed("upstream-import"):
        target_run.publish("upstream-import", [target],
                           [target_run.receipt_path(stage) for stage in UPSTREAM_STAGES],
                           metadata={"stages": list(UPSTREAM_STAGES)})
    return read_json(target)
