import copy
from pathlib import Path

import yaml

from prism.io import digest
from prism.v3.artifacts import Run
from prism.v5.config import load_config as load_v5_config
from prism.v6.bootstrap import UPSTREAM_STAGES, import_upstream
from prism.v6.config import load_config as load_v6_config


ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, source_name, target_name):
    source = yaml.safe_load((ROOT / "configs/experiment.v5.yaml").read_text(encoding="utf-8"))
    source["output"] = str(tmp_path / "source-run")
    source_path = tmp_path / source_name
    source_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    target = yaml.safe_load((ROOT / "configs/experiment.v6.yaml").read_text(encoding="utf-8"))
    target["output"] = str(tmp_path / "target-run")
    target_path = tmp_path / target_name
    target_path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")
    return source_path, target_path


def test_v6_imports_only_budget_independent_v5_assets_and_is_idempotent(tmp_path):
    source_path, target_path = _write(tmp_path, "v5.yaml", "v6.yaml")
    source = Run(load_v5_config(source_path))
    originals = {}
    for index, stage in enumerate(UPSTREAM_STAGES):
        artifact = source.file(f"upstream/{index:02d}-{stage}.bin")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"sealed-{stage}".encode())
        source.publish(stage, [artifact])
        originals[artifact.relative_to(source.path)] = digest(artifact)
    target_config = load_v6_config(target_path)
    first = import_upstream(target_config, source_path)
    second = import_upstream(target_config, source_path)
    assert first == second
    target = Run(target_config)
    target.require("upstream-import")
    assert set(first["stages"]) == set(UPSTREAM_STAGES)
    for relative, expected in originals.items():
        assert digest(target.file(relative)) == expected
    for forbidden in ("design-0.001", "design-report", "study-selection",
                      "sampling", "reward", "normalization", "policy-prism"):
        assert not target.receipt_path(forbidden).exists()
