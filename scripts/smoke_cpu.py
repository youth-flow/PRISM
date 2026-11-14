"""Minimal scheduled CPU/container/network probe; performs no experiment work."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import urllib.request


def main():
    expected = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if affinity is not None and affinity < expected:
        raise RuntimeError(f"CPU affinity {affinity} is smaller than Slurm allocation {expected}")
    tmp = Path(os.environ["TMPDIR"]).resolve()
    with tempfile.NamedTemporaryFile(dir=tmp, prefix="prism-cpu-smoke-", delete=True) as file:
        file.write(b"ok")
        file.flush()
    request = urllib.request.Request(
        "https://huggingface.co/api/models?limit=1",
        headers={"User-Agent": "PRISM-Yancheng-CPU-smoke"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200 or not response.read(64):
            raise RuntimeError(f"Hugging Face network probe failed with status {response.status}")
    packages = {}
    for name in ("numpy", "scipy", "torch", "transformers", "trl", "datasets", "nltk"):
        packages[name] = importlib.metadata.version(name)
    print(json.dumps({
        "host": platform.node(),
        "python": platform.python_version(),
        "slurm_cpus_per_task": expected,
        "affinity_cpus": affinity,
        "tmpdir": str(tmp),
        "proxy_configured": bool(os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")),
        "network_probe": "huggingface_ok",
        "packages": packages,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
