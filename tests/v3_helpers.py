"""Small genuine numerical V3 fixtures; no model, network or paid API."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from prism.io import write_rows
from prism.v3.artifacts import Run, write_json
from prism.v3.config import load_config
from prism.v3.reward import save_npz


def configuration(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((repo / "configs/experiment.v3.yaml").read_text(encoding="utf-8"))
    c["output"] = str(tmp_path / "run")
    c["runtime"]["device"] = "cpu"
    c["data"].update(raw_dir=str(tmp_path / "raw"), expected_train_rows=6, expected_test_rows=12,
                     policy_prompts=4, reward_dev_pairs=4, ppo_dev_prompts=2, evaluation_prompts=2,
                     feature_dimension=3, context_tokens=128, max_prompt_tokens=64)
    c["features"]["shards"] = 1
    c["reward"].update(feature_dimension=3, calibration_responses=8)
    c["ppo"].update(total_episodes=8, rollout_batch_size=4, mini_batch_size=2,
                    per_device_train_batch_size=1, gradient_accumulation_steps=2,
                    num_mini_batches=2, num_ppo_epochs=2, critic_warmup_episodes=4,
                    critic_warmup_epochs=2, critic_warmup_mini_batch_size=2,
                    response_length=8, engineering_checkpoint=4, checkpoints=[4, 8],
                    local_rollout_forward_batch_size=2, fixed_train_probe_prompts=2)
    c["evaluation"].update(max_new_tokens=8, bootstrap_samples=20)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    return load_config(path)


def numerical_run(c):
    run = Run(c)
    v = np.concatenate([np.eye(3), -np.eye(3)])
    ids = np.array([f"pair-{i}" for i in range(len(v))])
    pool = [{"id": str(pair), "prompt_id": f"context-{i}",
             "messages": [{"role": "user", "content": f"question {i}"}],
             "response_A": f"answer A {i}", "response_B": f"answer B {i}",
             "source": {"subset": "helpful-base", "split": "train", "line": i + 1}}
            for i, pair in enumerate(ids)]
    labels = [{"id": str(pair), "label": int(i < 3)} for i, pair in enumerate(ids)]
    dev_v = np.array([[1, 0.2, 0], [0, 1, 0.1], [-1, 0, 0], [0, 0, -1]], dtype=float)
    dev = [{"id": f"dev-{i}", "prompt_id": f"dev-context-{i}",
            "messages": [{"role": "user", "content": f"dev question {i}"}],
            "response_A": f"A {i}", "response_B": f"B {i}", "label": int(i < 2),
            "source": {"subset": "helpful-base", "split": "test", "line": i + 1}}
           for i in range(len(dev_v))]
    outputs = []
    for filename, values in (("pool.jsonl", pool), ("labels.jsonl", labels), ("reward_dev.jsonl", dev),
                             ("policy.jsonl", pool[:4]), ("ppo_dev.jsonl", dev[:2]), ("evaluation.jsonl", dev[2:])):
        path = run.file("data/" + filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_rows(path, values)
        outputs.append(path)
    run.publish("data", outputs)
    phi = np.stack([v / 2, -v / 2], axis=1).astype(np.float32)
    dev_phi = np.stack([dev_v / 2, -dev_v / 2], axis=1).astype(np.float32)
    for filename, pp, ii in (("pool.npz", phi, ids), ("reward_dev.npz", dev_phi, np.array([r["id"] for r in dev]))):
        save_npz(run.file("features/" + filename), phi=pp, ids=ii, logp=np.zeros(pp.shape[:2]))
    run.publish("features", [run.file("features/pool.npz"), run.file("features/reward_dev.npz")], [run.receipt_path("data")])
    save_npz(run.file("geometry/geometry.npz"), v=v, K=np.eye(3) / 12, b=np.zeros(3),
             proxy_probabilities=np.full((len(v), 2), 0.5), ids=ids, empirical=np.full(len(v), 1 / len(v)))
    run.publish("geometry", [run.file("geometry/geometry.npz")], [run.receipt_path("features")])
    from prism.v3.audit import audit_geometry
    audit_geometry(c)
    receipts = []
    for ridge in c["design"]["lambdas"]:
        directory = run.file(f"design/lambda-{float(ridge)}")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "q.npy", np.full(len(v), 1 / len(v)))
        write_json(directory / "report.json", {"converged": True})
        name = f"design-{float(ridge)}"
        run.publish(name, [directory / "q.npy", directory / "report.json"], [run.receipt_path("geometry")])
        receipts.append(run.receipt_path(name))
    write_json(run.file("design/report.json"), {"candidates": c["design"]["lambdas"]})
    run.publish("design-report", [run.file("design/report.json")], receipts)
    return run
