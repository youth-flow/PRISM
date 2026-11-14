"""Benchmark adapters; IFEval delegates to pinned, staged Google Research code.

Scoring and preflight are offline. Only ``stage_ifeval_sources`` downloads code,
and callers must expose that function through an explicit preparation stage.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import urllib.request

from .io import prompt_rows, rows


IFEVAL_SOURCE_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
IFEVAL_DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFEVAL_REPOSITORY = "https://github.com/google-research/google-research"
_SOURCE_FILES = (
    "instruction_following_eval/evaluation_lib.py",
    "instruction_following_eval/instructions_registry.py",
    "instruction_following_eval/instructions.py",
    "instruction_following_eval/instructions_util.py",
)
_KNOWN_HASHES = {
    _SOURCE_FILES[0]: "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    _SOURCE_FILES[1]: "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    _SOURCE_FILES[2]: "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    _SOURCE_FILES[3]: "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
}
_MANIFEST = "source-manifest.json"


def _kind(c):
    return c["choices"]["benchmark"].get("kind", "exact_match")


def exact_match(response, answer):
    return " ".join(response.strip().casefold().split()) == " ".join(answer.strip().casefold().split())


def benchmark_rows(c):
    """Read normalized generation records while preserving benchmark metadata."""
    choice = c["choices"]["benchmark"]
    kind = _kind(c)
    if kind == "exact_match":
        data = prompt_rows(choice["path"])
        if any(not isinstance(r.get("answer"), str) for r in data):
            raise ValueError("exact_match benchmark requires a string answer per record")
        return data
    if kind != "ifeval":
        raise ValueError(f"Unknown benchmark kind: {kind}")
    data = rows(choice["path"])
    if not data:
        raise ValueError("IFEval input is empty")
    normalized = []
    for r in data:
        if type(r.get("key")) is not int or not isinstance(r.get("prompt"), str) or not r["prompt"]:
            raise ValueError("IFEval records require integer key and nonempty original prompt")
        instructions, kwargs = r.get("instruction_id_list"), r.get("kwargs")
        if (not isinstance(instructions, list) or not instructions
                or any(not isinstance(i, str) or not i for i in instructions)
                or not isinstance(kwargs, list) or len(kwargs) != len(instructions)
                or any(not isinstance(k, dict) for k in kwargs)):
            raise ValueError("IFEval requires matched instruction_id_list and kwargs lists")
        messages = [{"role": "user", "content": r["prompt"]}]
        if "messages" in r and r["messages"] != messages:
            raise ValueError("IFEval messages must preserve the original prompt exactly")
        record_id = r.get("id", str(r["key"]))
        if not isinstance(record_id, (str, int)) or isinstance(record_id, bool):
            raise ValueError("Benchmark IDs must be strings or integers")
        normalized.append({**r, "id": record_id, "messages": messages})
    for field in ("id", "key", "prompt"):
        if len({r[field] for r in normalized}) != len(normalized):
            raise ValueError(f"Duplicate IFEval {field}; official checker keys responses by prompt")
    return normalized


def _verified_source_root(c):
    choice = c["choices"]["benchmark"]
    revision = choice.get("official_source_revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("IFEval official_source_revision must be a pinned 40-character commit")
    configured = choice.get("official_source_path")
    if not configured:
        raise RuntimeError("IFEval official_source_path is required; stage the official checker first")
    source = Path(configured).resolve()
    manifest_path = source / _MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"IFEval source is not staged at {source}; run the explicit source preparation stage")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != IFEVAL_REPOSITORY or manifest.get("revision") != revision:
        raise ValueError("IFEval source manifest does not match the configured official revision")
    files = manifest.get("files", {})
    for relative in (*_SOURCE_FILES, "LICENSE"):
        file = source / relative
        if not file.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(files.get(relative, ""))):
            raise ValueError(f"Missing pinned IFEval source artifact: {relative}")
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != files[relative]:
            raise ValueError(f"IFEval source hash changed: {relative}")
        if revision == IFEVAL_SOURCE_REVISION and relative in _KNOWN_HASHES and actual != _KNOWN_HASHES[relative]:
            raise ValueError(f"IFEval source differs from the verified official snapshot: {relative}")
    return source


@contextmanager
def _official_checker(c):
    """Isolate the official namespace so another checkout cannot shadow this one."""
    source = _verified_source_root(c)
    prefix = "instruction_following_eval"
    saved = {k: v for k, v in sys.modules.items() if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(source))
    nltk_paths = None
    try:
        try:
            lib = importlib.import_module(prefix + ".evaluation_lib")
            util = importlib.import_module(prefix + ".instructions_util")
        except ImportError as exc:
            raise RuntimeError(
                "Official IFEval dependencies are unavailable; install the benchmark extra "
                "(absl-py, langdetect, nltk, immutabledict) in the runtime image"
            ) from exc
        if not Path(lib.__file__).resolve().is_relative_to(source):
            raise RuntimeError("IFEval import resolved outside the configured official source")
        nltk_paths = list(util.nltk.data.path)
        configured = c.get("benchmark", {}).get("nltk_data_path", c["choices"]["benchmark"].get("nltk_data_path"))
        if configured:
            util.nltk.data.path.insert(0, str(Path(configured).resolve()))
        try:
            # Official checkers use both interfaces. Modern NLTK may require
            # punkt_tab as well as punkt; no download is attempted here.
            util.count_sentences("A sentence. A second sentence.")
            util.nltk.word_tokenize("An English phrase.")
        except LookupError as exc:
            raise RuntimeError(
                "IFEval NLTK resources are missing; explicitly stage punkt and punkt_tab "
                "and configure nltk_data_path or NLTK_DATA before GPU evaluation"
            ) from exc
        yield lib
    finally:
        if nltk_paths is not None:
            util.nltk.data.path[:] = nltk_paths
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
        else:
            sys.path.remove(str(source))
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(saved)


def _instruction_kwargs(record):
    # HF Dataset structs pad unused argument fields with null. Null padding is
    # not an extra keyword argument to every official checker constructor.
    return [{key: value for key, value in kwargs.items() if value is not None}
            for kwargs in record["kwargs"]]


def validate_benchmark_backend(c):
    """Fail before model loading if benchmark data/checker/resources are missing."""
    data = benchmark_rows(c)
    if _kind(c) == "exact_match":
        return {"kind": "exact_match", "count": len(data)}
    with _official_checker(c) as lib:
        registry = importlib.import_module("instruction_following_eval.instructions_registry")
        for r in data:
            for instruction_id, kwargs in zip(r["instruction_id_list"], _instruction_kwargs(r)):
                if instruction_id not in registry.INSTRUCTION_DICT:
                    raise ValueError(f"Unknown official IFEval instruction: {instruction_id}")
                try:
                    registry.INSTRUCTION_DICT[instruction_id](instruction_id).build_description(**kwargs)
                except (TypeError, ValueError, AssertionError) as exc:
                    raise ValueError(f"Invalid IFEval instruction arguments for key={r['key']}") from exc
        if not hasattr(lib, "test_instruction_following_strict"):
            raise RuntimeError("Configured IFEval source lacks the official strict checker")
    return {"kind": "ifeval", "count": len(data), "official_source_revision": c["choices"]["benchmark"]["official_source_revision"]}


def score_benchmark(c, response_records):
    """Score an exact set of saved responses; primary IFEval metric is prompt strict."""
    data = benchmark_rows(c)
    records = list(response_records)
    if any(not isinstance(r.get("response"), str) for r in records):
        raise ValueError("Benchmark responses must contain string response fields")
    ids = [r.get("id") for r in records]
    if len(set(ids)) != len(ids) or set(ids) != {r["id"] for r in data}:
        raise ValueError("Benchmark response IDs must exactly match input IDs, without duplicates")
    by_id = {r["id"]: r["response"] for r in records}
    if _kind(c) == "exact_match":
        details = [{"id": r["id"], "correct": exact_match(by_id[r["id"]], r["answer"])} for r in data]
        score = sum(r["correct"] for r in details)/len(details)
        return {"metric": "normalized_exact_match", "score": score, "metrics": {"normalized_exact_match": score}, "details": details}
    with _official_checker(c) as lib:
        prompt_to_response = {r["prompt"]: by_id[r["id"]] for r in data}
        strict, loose, details = [], [], []
        for r in data:
            inp = lib.InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"], prompt=r["prompt"], kwargs=_instruction_kwargs(r))
            s = lib.test_instruction_following_strict(inp, prompt_to_response)
            l = lib.test_instruction_following_loose(inp, prompt_to_response)
            strict.append(s)
            loose.append(l)
            details.append({"id": r["id"], "key": r["key"], "strict": asdict(s), "loose": asdict(l)})
    def accuracy(outputs, instruction_level):
        if instruction_level:
            flags = [flag for output in outputs for flag in output.follow_instruction_list]
        else:
            flags = [output.follow_all_instructions for output in outputs]
        return sum(flags)/len(flags)
    metrics = {"prompt_level_strict": accuracy(strict, False), "instruction_level_strict": accuracy(strict, True),
               "prompt_level_loose": accuracy(loose, False), "instruction_level_loose": accuracy(loose, True)}
    return {"metric": "prompt_level_strict", "score": metrics["prompt_level_strict"], "metrics": metrics,
            **metrics, "details": details, "official_source_revision": c["choices"]["benchmark"]["official_source_revision"]}


def _download_official(revision, relative):
    url = f"https://raw.githubusercontent.com/google-research/google-research/{revision}/{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": "PRISM-benchmark-preparation"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def stage_ifeval_sources(destination, revision=IFEVAL_SOURCE_REVISION):
    """Explicitly download a tiny official-source subset; never install packages."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("IFEval source staging requires an immutable 40-character revision")
    destination = Path(destination).resolve()
    c = {"choices": {"benchmark": {"kind": "ifeval", "official_source_path": str(destination), "official_source_revision": revision}}}
    if destination.exists():
        _verified_source_root(c)
        return json.loads((destination/_MANIFEST).read_text(encoding="utf-8"))
    paths = (*_SOURCE_FILES, "instruction_following_eval/README.md", "instruction_following_eval/requirements.txt", "LICENSE")
    artifacts = {relative: _download_official(revision, relative) for relative in paths}
    hashes = {relative: hashlib.sha256(content).hexdigest() for relative, content in artifacts.items()}
    if revision == IFEVAL_SOURCE_REVISION and any(hashes[name] != value for name, value in _KNOWN_HASHES.items()):
        raise RuntimeError("Downloaded IFEval source does not match verified official hashes")
    manifest = {"repository": IFEVAL_REPOSITORY, "revision": revision, "files": hashes,
                "note": "Official subset; original Apache license and source files preserved without edits."}
    staging = destination.with_name(destination.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    for relative, content in artifacts.items():
        target = staging/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as file:
            file.write(content)
    (staging/_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    return manifest


def stage_ifeval_resources(c):
    """Explicit preparation only: stage NLTK assets without changing global env."""
    if _kind(c) != "ifeval":
        return {"kind": _kind(c), "resources": []}
    configured = c.get("benchmark", {}).get("nltk_data_path", c["choices"]["benchmark"].get("nltk_data_path"))
    if not configured:
        raise ValueError("Configure benchmark.nltk_data_path before explicit IFEval resource staging")
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError("Install the benchmark extra before staging IFEval NLTK resources") from exc
    destination = Path(configured).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resources = ("punkt", "punkt_tab")
    manifest_path = destination / "resource-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("nltk_version") != nltk.__version__ or manifest.get("resources") != list(resources) or not manifest.get("files"):
            raise ValueError("Staged NLTK resource manifest does not match the runtime")
        for relative, expected in manifest["files"].items():
            # Older Windows staging manifests used backslashes. The actual
            # archive paths are portable; accept both separators when verifying.
            file = (destination / relative.replace("\\", "/")).resolve()
            if not file.is_relative_to(destination) or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Staged NLTK resource hash changed: {relative}")
        return manifest
    for resource in resources:
        if not nltk.download(resource, download_dir=str(destination), quiet=True, raise_on_error=True):
            raise RuntimeError(f"NLTK resource staging failed: {resource}")
    manifest = {"nltk_version": nltk.__version__, "resources": list(resources), "files": {
        file.relative_to(destination).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted((destination/"tokenizers").rglob("*")) if file.is_file()
    }}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest
