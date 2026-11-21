"""Hash-verified import of V5 assets that are independent of V6 Design."""
import os
import shutil

from prism.io import digest
from prism.v3.artifacts import Run, read_json, write_json
from prism.v3.config import load_config as load_v3_config
from prism.v5.config import validate_v5

from .config import validate_v6


UPSTREAM_STAGES = (
    "data", "features", "geometry", "geometry-audit", "shared-sft",
)
UNCHANGED_SECTIONS = (
    "model", "runtime", "study", "data", "features", "reward", "ppo",
    "evaluation",
)


def _materialize(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or digest(target) != digest(source):
            raise ValueError(f"Preserve different V6 upstream artifact: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if digest(target) != digest(source):
        target.unlink(missing_ok=True)
        raise ValueError(f"Imported V6 artifact hash differs: {target}")


def import_upstream(c, upstream_config):
    """Recertify V5 outputs unaffected by the capped feasible region.

    Design distributions and every label-dependent/downstream artifact are
    deliberately excluded because the cap depends on the V6 budget.
    """
    validate_v6(c)
    upstream = load_v3_config(upstream_config)
    validate_v5(upstream)
    mismatches = [name for name in UNCHANGED_SECTIONS
                  if c["_contract"][name] != upstream["_contract"][name]]
    old_design = upstream["_contract"]["design"]
    new_design = c["_contract"]["design"]
    stable_design = [
        key for key in old_design
        if key not in {"solver", "extra_constraints"}
        and old_design[key] != new_design.get(key)
    ]
    old_sampling = upstream["_contract"]["sampling"]
    new_sampling = c["_contract"]["sampling"]
    stable_sampling = [
        key for key in old_sampling
        if key != "protocol" and old_sampling[key] != new_sampling.get(key)
    ]
    if mismatches or stable_design or stable_sampling:
        raise ValueError("V6/upstream immutable sections differ: " + str({
            "sections": mismatches,
            "design": stable_design,
            "sampling": stable_sampling,
        }))

    source_run, target_run = Run(upstream, create=False), Run(c)
    if source_run.path == target_run.path:
        raise ValueError("V6 import needs a distinct output directory")
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
                raise ValueError(f"Upstream output changed during V6 import: {source}")
            target = target_run.file(key[4:])
            _materialize(source, target)
            outputs.append(target)
        if target_run.completed(stage):
            current = target_run.require(stage)
            expected_outputs = {target_run._key(path): digest(path) for path in outputs}
            if current["outputs"] != expected_outputs:
                raise ValueError(f"Existing V6 imported stage differs: {stage}")
        else:
            target_run.publish(stage, outputs, [source_receipt], metadata={
                "import_protocol": "v6_hash_verified_v5_upstream_v1",
                "upstream_config_sha256": source_run.fingerprint,
                "upstream_receipt_sha256": digest(source_receipt),
            })
        imported[stage] = {
            "files": len(outputs),
            "source_receipt_sha256": digest(source_receipt),
            "target_receipt_sha256": digest(target_run.receipt_path(stage)),
        }
    summary = {
        "protocol": "v6_hash_verified_v5_upstream_v1",
        "upstream_run": str(source_run.path),
        "target_run": str(target_run.path),
        "stages": imported,
        "excluded_as_budget_dependent": [
            "design-*", "design-report", "study-selection", "sampling",
            "reward", "training-audit", "normalization", "linear-ppo",
            "evaluation",
        ],
    }
    target = target_run.file("control/upstream-import.json")
    write_json(target, summary)
    if not target_run.completed("upstream-import"):
        target_run.publish(
            "upstream-import", [target],
            [target_run.receipt_path(stage) for stage in UPSTREAM_STAGES],
            metadata={"stages": list(UPSTREAM_STAGES)},
        )
    return read_json(target)
