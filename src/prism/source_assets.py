"""Stage pinned raw sources locally and read them offline on compute nodes."""
import argparse
import json
from pathlib import Path

from .io import digest, dump, load_config


def _expected(c):
    source, benchmark = c["data"]["source"], c["data"]["benchmark_source"]
    return [(source["id"], source["revision"], source[key], "parquet") for key in ("train_split", "test_split")] + [
        (benchmark["id"], benchmark["revision"], benchmark["split"], "json")]


def stage_sources(c, cache):
    from huggingface_hub import snapshot_download
    cache = Path(cache).resolve()
    sources = []
    for repo, revision, split, fmt in _expected(c):
        if fmt == "parquet":
            patterns = [f"data/{split}-*.parquet"]
        elif repo == "google/IFEval":
            patterns = ["ifeval_input_data.jsonl"]
        else:
            raise ValueError("The pinned benchmark source requires an explicit raw-file adapter")
        snapshot = Path(snapshot_download(repo, repo_type="dataset", revision=revision, cache_dir=str(cache / "hub"), allow_patterns=patterns))
        files = sorted(file for pattern in patterns for file in snapshot.glob(pattern))
        if not files:
            raise RuntimeError(f"No pinned raw source files for {repo}/{split}")
        sources.append({"id": repo, "revision": revision, "split": split, "format": fmt,
                        "files": [{"path": f.relative_to(cache).as_posix(), "sha256": digest(f)} for f in files]})
    path = cache / "sources.manifest.json"
    manifest = {"kind": "prism_raw_sources", "version": 1, "sources": sources}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise FileExistsError("Refusing to replace a different raw-source manifest")
    dump(path, manifest)
    return {"manifest": str(path), "sha256": digest(path), "splits": len(sources)}


def offline_loader(c, manifest_path):
    from datasets import load_dataset
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "prism_raw_sources" or manifest.get("version") != 1:
        raise ValueError("Raw-source manifest kind/version mismatch")
    sources = manifest["sources"]
    expected = set(_expected(c))
    if len(sources) != len(expected) or {(s["id"], s["revision"], s["split"], s["format"]) for s in sources} != expected:
        raise ValueError("Raw-source revisions/splits do not match the frozen experiment")
    verified = {}
    for source in sources:
        files = []
        for item in source["files"]:
            path = (manifest_path.parent / item["path"]).resolve()
            if not path.is_relative_to(manifest_path.parent) or digest(path) != item["sha256"]:
                raise ValueError("Raw-source path/hash mismatch")
            files.append(str(path))
        if not files or len(files) != len(set(files)):
            raise ValueError("Raw source needs distinct nonempty files")
        verified[(source["id"], source["revision"], source["split"])] = (source["format"], files)
    def load(repo, *, revision, split):
        fmt, files = verified[(repo, revision, split)]
        # The local file order equals the pinned repository's shard order.
        # Read only prompt/ID columns; existing preference data are never used.
        options = {}
        if fmt == "parquet":
            s = c["data"]["source"]
            options["columns"] = [s["prompt_field"], s["id_field"]]
        return load_dataset(fmt, data_files=files, split="train", **options)
    return load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    c, _ = load_config(args.config)
    print(json.dumps(stage_sources(c, args.cache), ensure_ascii=False))


if __name__ == "__main__":
    main()
