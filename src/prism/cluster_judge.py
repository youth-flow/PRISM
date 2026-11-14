"""Run only the DeepSeek training Judge on cluster CPU allocations."""
import argparse
import json
import os
from pathlib import Path
import stat

from .io import cell, load_config, verify_receipt
from .judges import judge_settings
from .local_judge import export_packet, import_results, run_packet, validate_packet


def run_training(c, method, budget, seed, *, workers=4, execute=False, transport=None):
    if not execute:
        raise RuntimeError("Cluster DeepSeek calls require explicit execution")
    if judge_settings(c, "training")["backend"] != "deepseek":
        raise ValueError("Cluster API execution is restricted to the DeepSeek training Judge")
    out = cell(c, method, budget, seed)
    marker = out / "labels.receipt.json"
    if marker.exists():
        return verify_receipt(marker)
    packet, results = out / "training.packet.json", out / "training.api.results.jsonl"
    if not packet.exists():
        export_packet(c, "label", method, budget, seed, packet)
    validate_packet(c, packet)  # Check immutable inputs before spending.
    try:
        run_packet(packet, results, execute=True, workers=workers, transport=transport)
    except BaseException:
        # Preserve partial calls in the canonical stage ledger as well. The
        # source results remain append-only even when the API is interrupted.
        try:
            import_results(c, packet, results)
        except RuntimeError:
            pass
        raise
    import_results(c, packet, results)
    return verify_receipt(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run the cluster Judge inside a CPU Slurm allocation")
    path = Path(args.key_file).resolve()
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError("DeepSeek key file must be owned by the current user with private permissions")
    c, _ = load_config(args.config, ready=True)
    settings = judge_settings(c, "training")
    os.environ[settings["api_key_env"]] = path.read_text(encoding="utf-8").strip()
    result = run_training(c, args.method, args.budget, args.seed, workers=args.workers, execute=args.execute)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
