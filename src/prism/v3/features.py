"""Existing HH answer scoring, without generation or backbone updates."""
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import time

import numpy as np

from prism.io import rows
from .artifacts import Run, hash_json, read_json, write_json
from .data import encode_pair, tokenizer_fingerprint
from .recovery import recover_stage, seal_and_publish, stage_lock


def _sources():
    folder = Path(__file__).resolve().parent
    return [Path(__file__), folder / "data.py", folder.parent / "hf.py", folder.parent / "numerics.py"]


def _backend_contract(scorer):
    if scorer is None:
        return {"backend": "frozen_hf_fp32"}
    function = scorer if inspect.isfunction(scorer) else type(scorer)
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = function.__module__ + ":" + function.__qualname__
    return {"backend": "injected_test_scorer", "code_sha256": hashlib.sha256(source.encode()).hexdigest()}


def _validate_shard(c, run, pool, dev, shard_index, shards):
    dimension = c["data"]["feature_dimension"]
    report = read_json(run.file(f"features/part-{shard_index:05d}-of-{shards:05d}.json"))
    expected = {"shard_index": shard_index, "shards": shards, "dimension": dimension,
                "model": c["model"], "eos_logp": "included_in_canonical_response",
                "generated_responses": 0, "backbone_updates": 0,
                "tokenizer_semantics_sha256": hash_json(read_json(run.file("data/manifest.json"))["tokenizer_semantics"])}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Recovered feature report model/protocol changed")
    with np.load(run.file(f"features/part-{shard_index:05d}-of-{shards:05d}.npz"), allow_pickle=False) as arrays:
        for records, prefix, report_key in ((pool, "", "pool_rows"), (dev, "dev_", "dev_rows")):
            indices = np.arange(shard_index, len(records), shards, dtype=np.int64)
            phi, logp = arrays[prefix + "phi"], arrays[prefix + "logp"]
            if (not np.array_equal(arrays[prefix + "indices"], indices)
                    or arrays[prefix + "indices"].dtype != np.int64
                    or arrays[prefix + "ids"].tolist() != [records[i]["id"] for i in indices]
                    or phi.shape != (len(indices), 2, dimension) or logp.shape != (len(indices), 2)
                    or phi.dtype != np.float32 or logp.dtype != np.float64
                    or not np.isfinite(phi).all() or not np.isfinite(logp).all()
                    or np.any(logp > 1e-6) or report.get(report_key) != len(indices)):
                raise ValueError("Recovered feature shard identity/numerics mismatch")
    return report


def model_contract(c):
    return {"choices": {"sft": {"model": c["model"]["id"], "revision": c["model"]["revision"]}},
            "runtime": c["runtime"]}


class HFPairScorer:
    """The same frozen EOS map used for both human and generated answers."""
    def __init__(self, c):
        from prism.hf import load_lm, tokenizer
        self.c = c
        self.tokenizer = tokenizer(model_contract(c))
        manifest = Run(c).file("data/manifest.json")
        if manifest.exists() and read_json(manifest).get("tokenizer_semantics") != tokenizer_fingerprint(self.tokenizer):
            raise ValueError("Feature tokenizer semantics differ from prepared HH inputs")
        self.model = load_lm(model_contract(c))
        if any(p.requires_grad for p in self.model.parameters()):
            self.model.requires_grad_(False)
        if any(p.is_floating_point() and str(p.dtype) != "torch.float32" for p in self.model.parameters()):
            raise ValueError("V3 feature model must use FP32 parameters")
        self._score = lru_cache(maxsize=65536)(self._score_uncached)

    def _score_uncached(self, prompt, response):
        from prism.hf import sequence_stats
        return sequence_stats(self.model, list(prompt), list(response), self.tokenizer.eos_token_id)

    def __call__(self, record):
        prompt, responses = encode_pair(record, self.tokenizer, self.c["data"]["context_tokens"])
        scored = [self._score(tuple(prompt), tuple(response)) for response in responses]
        return np.stack([r[0] for r in scored]), np.asarray([r[1] for r in scored])


def _arrays(records, indices, scorer, dimension):
    phi = np.empty((len(indices), 2, dimension), dtype=np.float32)
    logp = np.empty((len(indices), 2), dtype=np.float64)
    for j, index in enumerate(indices):
        features, probabilities = map(np.asarray, scorer(records[index]))
        if (features.shape != (2, dimension) or probabilities.shape != (2,)
                or not np.isfinite(features).all() or not np.isfinite(probabilities).all()
                or np.any(probabilities > 1e-6)):
            raise ValueError(f"Invalid human-answer feature/logp at {records[index]['id']}")
        phi[j], logp[j] = features, probabilities
    return {"indices": np.asarray(indices, dtype=np.int64), "phi": phi, "logp": logp,
            "ids": np.asarray([records[i]["id"] for i in indices], dtype="U64")}


def extract_features(c, shard_index=0, shards=1, *, scorer=None):
    run = Run(c)
    stage = f"features-part-{shard_index:05d}-of-{shards:05d}"
    with stage_lock(run, stage):
        return _extract_features(c, shard_index, shards, scorer=scorer)


def _extract_features(c, shard_index=0, shards=1, *, scorer=None):
    run = Run(c)
    run.require("data")
    pool, dev = rows(run.file("data/pool.jsonl")), rows(run.file("data/reward_dev.jsonl"))
    if (type(shards) is not int or not 1 <= shards <= len(pool)
            or type(shard_index) is not int or not 0 <= shard_index < shards):
        raise ValueError("Feature shards need 0 <= index < shards <= pool size")
    stage = f"features-part-{shard_index:05d}-of-{shards:05d}"
    if run.completed(stage):
        return read_json(run.file(f"features/part-{shard_index:05d}-of-{shards:05d}.json"))
    path = run.file(f"features/part-{shard_index:05d}-of-{shards:05d}.npz")
    marker = path.with_suffix(".json")
    dependencies = [run.receipt_path("data")]
    backend = _backend_contract(scorer)
    if recover_stage(run, stage, [path, marker], dependencies, code_paths=_sources(), context=backend,
                     validator=lambda: _validate_shard(c, run, pool, dev, shard_index, shards)):
        return read_json(marker)
    scorer = HFPairScorer(c) if scorer is None else scorer
    started = time.monotonic()
    dimension = c["data"]["feature_dimension"]
    pool_arrays = _arrays(pool, list(range(shard_index, len(pool), shards)), scorer, dimension)
    dev_arrays = _arrays(dev, list(range(shard_index, len(dev), shards)), scorer, dimension)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **pool_arrays, **{f"dev_{k}": v for k, v in dev_arrays.items()})
    report = {"shard_index": shard_index, "shards": shards, "pool_rows": len(pool_arrays["ids"]),
              "dev_rows": len(dev_arrays["ids"]), "dimension": dimension,
              "elapsed_seconds": time.monotonic() - started, "generated_responses": 0,
              "backbone_updates": 0, "model": c["model"], "eos_logp": "included_in_canonical_response",
              "tokenizer_semantics_sha256": hash_json(read_json(run.file("data/manifest.json"))["tokenizer_semantics"])}
    write_json(marker, report)
    seal_and_publish(run, stage, [path, marker], dependencies, metadata=report, code_paths=_sources(), context=backend)
    return report


def merge_features(c, shards):
    with stage_lock(Run(c), "features"):
        return _merge_features(c, shards)


def _merge_features(c, shards):
    run = Run(c)
    run.require("data")
    if run.completed("features"):
        return read_json(run.file("features/manifest.json"))
    pool, dev = rows(run.file("data/pool.jsonl")), rows(run.file("data/reward_dev.jsonl"))
    if type(shards) is not int or not 1 <= shards <= len(pool):
        raise ValueError("Invalid number of feature shards")
    dimension = c["data"]["feature_dimension"]
    dependencies = [run.receipt_path("data")]
    for shard in range(shards):
        stage = f"features-part-{shard:05d}-of-{shards:05d}"
        run.require(stage)
        dependencies.append(run.receipt_path(stage))
    outputs = [run.file("features/pool.npz"), run.file("features/reward_dev.npz"), run.file("features/manifest.json")]
    if recover_stage(run, "features", outputs, dependencies, code_paths=_sources(), context={"shards": shards}):
        return read_json(run.file("features/manifest.json"))
    combined = {}
    covered = {}
    for name, records in (("pool", pool), ("dev", dev)):
        combined[name] = {"phi": np.empty((len(records), 2, dimension), dtype=np.float32),
                          "logp": np.empty((len(records), 2), dtype=np.float64),
                          "ids": np.asarray([r["id"] for r in records], dtype="U64")}
        covered[name] = np.zeros(len(records), dtype=bool)
    for shard in range(shards):
        stage = f"features-part-{shard:05d}-of-{shards:05d}"
        run.require(stage)
        path = run.file(f"features/part-{shard:05d}-of-{shards:05d}.npz")
        with np.load(path, allow_pickle=False) as arrays:
            for name, records, prefix in (("pool", pool, ""), ("dev", dev, "dev_")):
                expected = np.arange(shard, len(records), shards, dtype=np.int64)
                indices = arrays[prefix + "indices"]
                ids = arrays[prefix + "ids"]
                phi, logp = arrays[prefix + "phi"], arrays[prefix + "logp"]
                if (not np.array_equal(indices, expected) or covered[name][indices].any()
                        or not np.array_equal(ids, combined[name]["ids"][indices])
                        or phi.shape != (len(indices), 2, dimension) or logp.shape != (len(indices), 2)
                        or not np.isfinite(phi).all() or not np.isfinite(logp).all()):
                    raise ValueError("Feature shard identity, coverage or numeric mismatch")
                combined[name]["phi"][indices] = phi
                combined[name]["logp"][indices] = logp
                covered[name][indices] = True
    if any(not coverage.all() for coverage in covered.values()):
        raise ValueError("Feature shard merge has missing rows")
    outputs = []
    for name, filename in (("pool", "pool.npz"), ("dev", "reward_dev.npz")):
        path = run.file(f"features/{filename}")
        with path.open("xb") as stream:
            np.savez(stream, **combined[name])
        outputs.append(path)
    report = {"pool_rows": len(pool), "reward_dev_rows": len(dev), "dimension": dimension,
              "shards": shards, "full_coverage": True, "pool_order_sha256": hash_json(combined["pool"]["ids"].tolist()),
              "generated_responses": 0, "backbone_updates": 0, "dtype": "float32",
              "matrix_dtype": "float64", "eos_logp": "included_in_canonical_response"}
    write_json(run.file("features/manifest.json"), report)
    outputs.append(run.file("features/manifest.json"))
    seal_and_publish(run, "features", outputs, dependencies, metadata=report, code_paths=_sources(), context={"shards": shards})
    return report
