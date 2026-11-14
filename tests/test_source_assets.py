import json
from pathlib import Path

import pytest

from prism.io import digest, dump
from prism.source_assets import offline_loader


def fixture(tmp_path):
    training, benchmark = tmp_path / "train.parquet", tmp_path / "benchmark.jsonl"
    training.write_bytes(b"test-only-source")
    benchmark.write_bytes(b"test-only-source")
    c = {"data": {"source": {"id": "source", "revision": "a" * 40, "train_split": "train_gen", "test_split": "test_gen", "prompt_field": "prompt", "id_field": "prompt_id"},
                  "benchmark_source": {"id": "google/IFEval", "revision": "b" * 40, "split": "train"}}}
    sources = [{"id": "source", "revision": "a" * 40, "split": split, "format": "parquet", "files": [{"path": training.name, "sha256": digest(training)}]} for split in ("train_gen", "test_gen")]
    sources.append({"id": "google/IFEval", "revision": "b" * 40, "split": "train", "format": "json", "files": [{"path": benchmark.name, "sha256": digest(benchmark)}]})
    manifest = tmp_path / "sources.manifest.json"
    dump(manifest, {"kind": "prism_raw_sources", "version": 1, "sources": sources})
    return c, manifest


def test_offline_sources_bind_revisions_hashes_and_prompt_only_columns(tmp_path, monkeypatch):
    import datasets
    c, manifest = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or [])
    load = offline_loader(c, manifest)
    load("source", revision="a" * 40, split="train_gen")
    assert calls[0][0] == ("parquet",)
    assert calls[0][1]["columns"] == ["prompt", "prompt_id"]
    assert Path(calls[0][1]["data_files"][0]).is_absolute()
    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed["sources"][0]["revision"] = "c" * 40
    dump(manifest, changed)
    with pytest.raises(ValueError, match="revisions/splits"):
        offline_loader(c, manifest)
    assert len(calls) == 1


def test_raw_source_tampering_fails_before_dataset_loading(tmp_path):
    c, manifest = fixture(tmp_path)
    (tmp_path / "train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="path/hash"):
        offline_loader(c, manifest)
