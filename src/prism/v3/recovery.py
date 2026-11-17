"""Recover sealed stage commits, or preserve incomplete attempts before retry.

Seals are written only after all output files have been closed. They bind the
same immutable config, input and output bytes as receipts, plus executor code.
A missing receipt never means permission to overwrite an existing result.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import uuid

from prism.io import digest
from .artifacts import read_json, write_json


def executor_hashes(paths):
    root = Path(__file__).resolve().parents[2]
    result = {}
    for path in [Path(__file__).resolve(), *map(Path, paths)]:
        absolute = path.resolve()
        name = absolute.relative_to(root).as_posix() if absolute.is_relative_to(root) else str(absolute)
        result[name] = digest(absolute)
    return result


def _table(run, paths):
    return {run._key(path): digest(path) for path in paths}


def seal_path(run, stage):
    return run.file(f"recovery/{stage}.seal.json")


@contextmanager
def stage_lock(run, stage):
    from filelock import FileLock
    path = run.file(f"locks/{stage}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path), timeout=0):
        yield


def seal_and_publish(run, stage, outputs, inputs=(), metadata=None, *, code_paths=(), context=None):
    value = {"protocol": "v3_stage_seal_v1", "stage": stage, "config_sha256": run.fingerprint,
             "inputs": _table(run, inputs), "outputs": _table(run, outputs),
             "executor_sources": executor_hashes(code_paths), "context": context or {}, "metadata": metadata or {}}
    write_json(seal_path(run, stage), value)
    return run.publish(stage, outputs, inputs, metadata=metadata)


def _archive(run, stage, paths, reason):
    sources = [Path(p).resolve() for p in paths if Path(p).is_file()]
    if not sources:
        return
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    archive = run.file(f"attempts/{stage}/{token}")
    entries = []
    for path in sources:
        if not path.is_relative_to(run.path):
            raise ValueError("Recovery cannot move a file outside the active run")
        relative = path.relative_to(run.path)
        target = (archive / relative).resolve()
        if not target.is_relative_to(archive) or not target.is_relative_to(run.path):
            raise ValueError("Unsafe attempt archive target")
        entries.append({"original": relative.as_posix(), "archived": target.relative_to(run.path).as_posix(),
                        "sha256": digest(path), "bytes": path.stat().st_size})
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive / "manifest.json", {"protocol": "v3_preserved_failed_attempt_v1", "stage": stage,
               "reason": reason, "config_sha256": run.fingerprint, "files": entries})
    for entry in entries:
        source, target = run.file(entry["original"]), run.file(entry["archived"])
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        if digest(target) != entry["sha256"]:
            raise RuntimeError("Attempt archive changed source bytes")


def recover_stage(run, stage, outputs, inputs=(), *, code_paths=(), validator=None, context=None):
    """Return True only after validating and publishing a complete sealed result.

    Call while holding the stage lock. Unsealed/untrusted output files are
    preserved with hashes, then removed from the final namespace by rename.
    Input verification errors propagate rather than archiving valid outputs.
    """
    inputs = list(map(Path, inputs))
    input_table = _table(run, inputs)
    marker = seal_path(run, stage)
    existing = [Path(p) for p in outputs if Path(p).exists()]
    if marker.exists():
        try:
            value = read_json(marker)
            expected = {"protocol": "v3_stage_seal_v1", "stage": stage,
                        "config_sha256": run.fingerprint, "inputs": input_table,
                        "executor_sources": executor_hashes(code_paths), "context": context or {}}
            if any(value.get(k) != v for k, v in expected.items()):
                raise ValueError("Unpublished stage seal contract/input/code changed")
            if value.get("outputs") != _table(run, outputs):
                raise ValueError("Unpublished stage output bytes changed")
            if validator is not None:
                validator()
        except (ValueError, TypeError, KeyError, OSError) as exc:
            _archive(run, stage, [*existing, marker], type(exc).__name__ + ": " + str(exc))
            return False
        # Do not catch a new receipt-publication interruption as an invalid
        # artifact: the valid seal must stay in place for the next recovery.
        run.publish(stage, outputs, inputs, metadata=value["metadata"])
        return True
    if existing:
        _archive(run, stage, existing, "Incomplete/unsealed stage output; recompute from fixed inputs")
    return False
