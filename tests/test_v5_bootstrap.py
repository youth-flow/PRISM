import copy
from pathlib import Path

import yaml

from prism.io import digest
from prism.v3.artifacts import Run
from prism.v3.config import load_config as load_v3_config
from prism.v5.bootstrap import UPSTREAM_STAGES, import_upstream
from prism.v5.config import load_config as load_v5_config


ROOT = Path(__file__).resolve().parents[1]


def _configs(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/experiment.v5.8k.yaml").read_text(encoding="utf-8"))
    raw["data"]["raw_dir"] = str(tmp_path / "raw")
    source = copy.deepcopy(raw)
    source.pop("v5")
    source["output"] = str(tmp_path / "source")
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    target = copy.deepcopy(raw)
    target["output"] = str(tmp_path / "target")
    target_path = tmp_path / "target.yaml"
    target_path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")
    return source_path, target_path


def test_v5_import_recertifies_exact_upstream_outputs_and_is_idempotent(tmp_path):
    source_path, target_path = _configs(tmp_path)
    source = Run(load_v3_config(source_path))
    originals = {}
    for index, stage in enumerate(UPSTREAM_STAGES):
        artifact = source.file(f"upstream/{index:02d}-{stage}.bin")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"sealed-{stage}".encode())
        source.publish(stage, [artifact])
        originals[artifact.relative_to(source.path)] = digest(artifact)
    target_config = load_v5_config(target_path)
    first = import_upstream(target_config, source_path)
    second = import_upstream(target_config, source_path)
    assert first == second
    target = Run(target_config)
    target.require("upstream-import")
    assert set(first["stages"]) == set(UPSTREAM_STAGES)
    for relative, expected in originals.items():
        assert digest(target.file(relative)) == expected
    assert not target.file("study/selection.json").exists()
    assert not target.file("reward/selection.json").exists()
