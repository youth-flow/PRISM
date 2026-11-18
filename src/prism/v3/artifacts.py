"""Portable hash-bound V3 stage receipts, independent of the V2 ledger."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from prism.io import digest


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if read_json(path) == value:
            return
        raise FileExistsError(f"Preserve existing V3 artifact: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def config_hash(config):
    contract = config.get("_contract")
    if contract is None:
        contract = {k: deepcopy(v) for k, v in config.items() if not k.startswith("_")}
    return hash_json(contract)


class Run:
    def __init__(self, config, *, create=True):
        self.config = config
        self.path = Path(config["output"]).resolve()
        self.fingerprint = config_hash(config)
        self._contract = config.get("_contract", {k: v for k, v in config.items() if not k.startswith("_")})
        protocol = config.get("protocol", "hh_finite_pool_v3")
        if protocol not in ("hh_finite_pool_v3", "hh_finite_pool_v4"):
            raise ValueError("Unknown HH experiment protocol")
        marker = self.path / "experiment.json"
        if marker.exists():
            existing = read_json(marker)
            if (existing.get("protocol") != protocol
                    or existing.get("config_sha256") != self.fingerprint
                    or hash_json(existing.get("config")) != self.fingerprint):
                raise ValueError("V3 output already belongs to a different experiment contract")
        elif create:
            write_json(marker, {"protocol": protocol, "config_sha256": self.fingerprint,
                                "config": self._contract})

    def file(self, relative):
        target = (self.path / relative).resolve()
        if not target.is_relative_to(self.path):
            raise ValueError("V3 artifact path must stay inside this run")
        return target

    def receipt_path(self, stage):
        if not isinstance(stage, str) or not stage or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in stage):
            raise ValueError("Invalid V3 stage name")
        return self.file(f"receipts/{stage}.json")

    def _key(self, path):
        value = Path(path).resolve()
        if value.is_relative_to(self.path):
            return "run:" + value.relative_to(self.path).as_posix()
        raw_dir = Path(self.config.get("data", {}).get("raw_dir", self.path / "__no_raw_source__")).resolve()
        if value.is_relative_to(raw_dir):
            return "raw:" + value.relative_to(raw_dir).as_posix()
        return "external:" + str(value)

    def _resolve(self, key):
        if key.startswith("run:"):
            return self.file(key[4:])
        if key.startswith("raw:"):
            base = Path(self.config["data"]["raw_dir"]).resolve()
            target = (base / key[4:]).resolve()
            if not target.is_relative_to(base):
                raise ValueError("Raw input dependency escapes its source directory")
            return target
        if key.startswith("external:"):
            return Path(key[9:])
        raise ValueError("Invalid V3 receipt path encoding")

    def publish(self, stage, outputs, inputs=(), metadata=None):
        output_paths = [Path(p).resolve() for p in outputs]
        if not output_paths or len(set(output_paths)) != len(output_paths):
            raise ValueError("A V3 receipt needs distinct output files")
        if any(not p.is_relative_to(self.path) or not p.is_file() for p in output_paths):
            raise ValueError("V3 output files must exist inside this run")
        input_paths = [Path(p).resolve() for p in inputs]
        if any(not p.is_file() for p in input_paths):
            raise ValueError("V3 input dependency does not exist")
        value = {"protocol": "v3_artifact_receipt_v1", "stage": stage,
                 "config_sha256": self.fingerprint,
                 "outputs": {self._key(p): digest(p) for p in output_paths},
                 "inputs": {self._key(p): digest(p) for p in input_paths},
                 "metadata": metadata or {}}
        write_json(self.receipt_path(stage), value)
        return self.require(stage)

    def require(self, stage, _visited=None, _verified=None, _digests=None):
        verified = {} if _verified is None else _verified
        digests = {} if _digests is None else _digests
        if stage in verified:
            return verified[stage]
        marker = self.receipt_path(stage)
        value = read_json(marker)
        if (value.get("protocol") != "v3_artifact_receipt_v1" or value.get("stage") != stage
                or value.get("config_sha256") != self.fingerprint):
            raise ValueError(f"V3 receipt contract mismatch: {stage}")
        visited = set() if _visited is None else _visited
        if stage in visited:
            raise ValueError("V3 receipt dependency cycle")
        visited.add(stage)
        for group in ("outputs", "inputs"):
            if not isinstance(value.get(group), dict) or (group == "outputs" and not value[group]):
                raise ValueError("Invalid V3 receipt file table")
            for key, expected in value[group].items():
                path = self._resolve(key)
                if path not in digests:
                    digests[path] = digest(path) if path.is_file() else None
                if digests[path] != expected:
                    raise ValueError(f"V3 artifact changed or missing: {path}")
                if group == "inputs" and path.parent == self.file("receipts") and path.suffix == ".json":
                    self.require(path.stem, visited, verified, digests)
        visited.remove(stage)
        verified[stage] = value
        return value

    def completed(self, stage):
        if not self.receipt_path(stage).exists():
            return False
        self.require(stage)
        return True

    def selection(self):
        self.require("study-selection")
        value = read_json(self.file("study/selection.json"))
        if value.get("budget_authority") != "user_after_design_report":
            raise ValueError("V3 budget requires an explicit user decision after Design")
        if type(value.get("budget")) is not int or value["budget"] < 1:
            raise ValueError("V3 preference budget has not been selected")
        return value
