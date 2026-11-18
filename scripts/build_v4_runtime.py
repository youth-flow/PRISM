"""Install offline wheels in a new private layer, from an allocated CPU job.

Call inside the verified SIF with --wheels and --output under online1. Package
resolution/download happens separately; every input wheel must match the
operator's wheels.sha256. Neither the old SIF nor an existing layer is edited.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def build(wheels, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Build V4 runtime in a Slurm CPU allocation")
    root = Path(os.environ["PRISM_STORAGE"]).resolve()
    wheels, output = Path(wheels).resolve(), Path(output).resolve()
    if not wheels.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise ValueError("Wheel inputs and the new layer must stay under online1")
    if output.exists():
        raise FileExistsError("Preserve the existing layer; choose a fresh immutable path")
    manifest = wheels / "wheels.sha256"
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = (wheels / name).resolve()
        if path.parent != wheels or path.suffix != ".whl" or name in entries:
            raise ValueError("Wheel manifest path/duplicate invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Wheel input hash changed")
        entries[name] = expected
    if not entries or {p.name for p in wheels.glob("*.whl")} != set(entries):
        raise ValueError("Every wheel must be sealed in the input manifest")
    lock = Path(__file__).resolve().parents[1] / "containers/requirements.v4-overlay.lock"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                    "--find-links", str(wheels), "--target", str(output), "-r", str(lock)], check=True)
    # Require imports and dependency constraints, before sealing the layer.
    env = {**os.environ, "PYTHONPATH": str(output), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-m", "pip", "check"], env=env, check=True)
    subprocess.run([sys.executable, "-c", "from transformers import Olmo3Config, Olmo3ForCausalLM; import peft"], env=env, check=True)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output).as_posix()}\n"
             for p in sorted(output.rglob("*")) if p.is_file()]
    seal = output / "runtime.sha256"
    seal.write_text("".join(lines), encoding="utf-8")
    print("V4_RUNTIME_SHA256=" + hashlib.sha256(seal.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.wheels, args.output)
